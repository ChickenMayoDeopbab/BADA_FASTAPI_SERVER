#!/usr/bin/env bash
# Qwen3-TTS 워커를 GPU 별로 여러 개 띄운다 (계획 0049).
#
# 동시성은 세마포어가 아니라 프로세스 수로 얻는다 — 한 프로세스 안에서 모델을 공유하면
# CUDA 그래프 버퍼가 충돌해 프로세스가 abort 한다(계획 0047). GPU 당 2개가 실측 상한이다
# (계획 0048: N=3 은 RTF 0.83, N=4 는 1.05~1.13 으로 실시간 불가).
#
# 사용법:
#   launch_workers.sh [GPUS] [PER_GPU] [BASE_PORT]   워커 기동 후 URL 목록 출력
#   launch_workers.sh stop                            띄운 워커 전부 종료
#
#   GPUS      물리 GPU 인덱스 콤마 목록 (기본 "1,2") — 승인 범위가 바뀌면 이 인자만 바꾼다
#   PER_GPU   GPU 당 워커 수 (기본 2)
#   BASE_PORT 첫 워커 포트 (기본 8010)
#
# 필요한 환경변수:
#   VOICES_FILE  voices.json 경로 (필수)
#   CC / CXX     torch.compile 이 런타임에 C 컴파일러를 부른다. 없으면 기동이 실패한다
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${QWEN_RUN_DIR:-$HOME/bada-qwen3-tts/run}"
PID_DIR="$RUN_DIR/pid"
LOG_DIR="$RUN_DIR/log"

stop_all() {
  if [ ! -d "$PID_DIR" ]; then
    echo "띄운 워커 없음 ($PID_DIR)"
    return 0
  fi
  for f in "$PID_DIR"/*.pid; do
    [ -e "$f" ] || continue
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "종료 $(basename "$f" .pid) (PID $pid)"
    else
      echo "이미 죽음 $(basename "$f" .pid) (PID $pid)"
    fi
    rm -f "$f"
  done
}

if [ "${1:-}" = "stop" ]; then
  stop_all
  exit 0
fi

GPUS="${1:-1,2}"
PER_GPU="${2:-2}"
BASE_PORT="${3:-8010}"

: "${VOICES_FILE:?VOICES_FILE 을 설정해야 한다 (voices.json 경로)}"
[ -f "$VOICES_FILE" ] || { echo "voices.json 이 없다: $VOICES_FILE" >&2; exit 1; }

PYTHON="${QWEN_PYTHON:-$ROOT/../../.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
SERVER_DIR="${QWEN_SERVER_DIR:-$ROOT}"
[ -f "$SERVER_DIR/server.py" ] || { echo "server.py 를 못 찾았다: $SERVER_DIR" >&2; exit 1; }

if [ -z "${CC:-}" ]; then
  echo "경고: CC 가 비어 있다. torch.compile 이 런타임에 C 컴파일러를 부르므로" >&2
  echo "      컴파일러가 PATH 에 없으면 기동이 InductorError 로 실패한다." >&2
fi

mkdir -p "$PID_DIR" "$LOG_DIR"
port=$BASE_PORT
urls=""

for gpu in ${GPUS//,/ }; do
  for i in $(seq 1 "$PER_GPU"); do
    name="gpu${gpu}-w${i}-p${port}"
    if [ -f "$PID_DIR/$name.pid" ] && kill -0 "$(cat "$PID_DIR/$name.pid")" 2>/dev/null; then
      echo "이미 떠 있음 $name — 건너뜀"
    else
      ( cd "$SERVER_DIR" && \
        CUDA_VISIBLE_DEVICES="$gpu" TTS_DEVICE="cuda:0" VOICES_FILE="$VOICES_FILE" \
        nohup "$PYTHON" -m uvicorn server:app --host 127.0.0.1 --port "$port" \
          > "$LOG_DIR/$name.log" 2>&1 & echo $! > "$PID_DIR/$name.pid" )
      echo "기동 $name (PID $(cat "$PID_DIR/$name.pid")) → $LOG_DIR/$name.log"
    fi
    urls="${urls:+$urls,}http://127.0.0.1:$port"
    port=$((port + 1))
  done
done

echo
echo "워밍업에 워커당 ~140초 걸린다. 준비 확인:"
echo "  for p in \$(seq $BASE_PORT $((port - 1))); do printf '%s ' \$p; curl -s -m 2 localhost:\$p/health; echo; done"
echo
echo "앱 설정에 넣을 값:"
echo "  QWEN_TTS_URLS=$urls"
