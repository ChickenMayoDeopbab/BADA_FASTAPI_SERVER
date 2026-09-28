#!/usr/bin/env bash
# CSM 워커를 GPU 별로 띄운다 — Qwen 의 launch_workers.sh 와 같은 꼴(설계 16 §2.4).
#   launch_csm_workers.sh [GPUS] [PER_GPU] [BASE_PORT]    기본 "0,3" 1 8020        · launch_csm_workers.sh stop
# 환경: CSM_DIR(워커 코드, 기본 ~/CSM) · CSM_PYTHON(기본 ~/csm-venv/bin/python) · CSM_WEIGHTS(기본 ~/CSM/runs/b1/epoch_1) · VOICES_FILE(필수) · CC/CXX(torch.compile)
set -euo pipefail
CSM_DIR="${CSM_DIR:-$HOME/CSM}"; PY="${CSM_PYTHON:-$HOME/csm-venv/bin/python}"; RUN="${CSM_RUN_DIR:-$HOME/CSM/run}"; mkdir -p "$RUN/pid" "$RUN/log"
if [ "${1:-}" = "stop" ]; then
  for f in "$RUN"/pid/*.pid; do [ -f "$f" ] || continue; p=$(cat "$f"); kill "$p" 2>/dev/null && echo "stop $(basename "$f" .pid) ($p)"; rm -f "$f"; done; exit 0
fi
GPUS="${1:-0,3}"; PER="${2:-1}"; PORT="${3:-8020}"
[ -n "${VOICES_FILE:-}" ] || { echo "VOICES_FILE 이 필요하다(voices.json)" >&2; exit 1; }
[ -n "${CC:-}" ] || echo "경고: CC 가 비어 있다 — torch.compile 이 C 컴파일러를 부른다(Qwen 과 같은 ~/gcc-env 사용)" >&2
urls=""
for gpu in ${GPUS//,/ }; do
  for i in $(seq 1 "$PER"); do
    name="csm-gpu${gpu}-w${i}-p${PORT}"
    if [ -f "$RUN/pid/$name.pid" ] && kill -0 "$(cat "$RUN/pid/$name.pid")" 2>/dev/null; then echo "이미 떠 있음 $name"; else
      ( cd "$CSM_DIR" && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" CSM_DEVICE="cuda" CSM_WEIGHTS="${CSM_WEIGHTS:-$HOME/CSM/runs/b1/epoch_1}" VOICES_FILE="$VOICES_FILE" \
        nohup "$PY" -m uvicorn server:app --host 127.0.0.1 --port "$PORT" > "$RUN/log/$name.log" 2>&1 & echo $! > "$RUN/pid/$name.pid" )
      echo "기동 $name (PID $(cat "$RUN/pid/$name.pid")) → $RUN/log/$name.log"
    fi
    urls="${urls:+$urls,}http://127.0.0.1:$PORT"; PORT=$((PORT + 1))
  done
done
echo; echo "워밍업(모델 로드 + 컴파일)에 워커당 2~3분(추정). 준비 확인:"; echo "  for u in ${urls//,/ }; do printf '%s ' \$u; curl -s -m 2 \$u/health; echo; done"; echo "앱 설정: CSM_TTS_URLS=$urls"
