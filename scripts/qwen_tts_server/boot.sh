#!/usr/bin/env bash
# 학교 GPU 서버 재시작 후 복구를 한 번에 (계획 0049).
#
# 컨테이너가 예고 없이 재시작되면 tailscaled 와 워커가 함께 죽는다. 그러면 EC2 가 워커에
# 닿지 못하고, 앱은 통화마다 헬스체크 1초를 버린 뒤 ElevenLabs 로 폴백한다(기능은 유지).
# 이 스크립트는 tailscaled → 로그인 확인 → 워커 기동 순서로 복구한다. 이미 떠 있는 건 건너뛴다.
#
# 사용법:
#   boot.sh [GPUS] [PER_GPU] [BASE_PORT]      기본 "1,2" 2 8010
#
# 최초 1회는 로그인이 필요하다 (auth key 는 Tailscale 콘솔에서 발급):
#   ~/bin/tailscale --socket=$HOME/.tailscale/tailscaled.sock up --authkey=tskey-... --hostname=school-gpu
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_DIR="$HOME/.tailscale"
TS_SOCK="$TS_DIR/tailscaled.sock"
TS="$HOME/bin/tailscale"
TSD="$HOME/bin/tailscaled"

ts() { "$TS" --socket="$TS_SOCK" "$@"; }

echo "== 1/3 tailscaled =="
if pgrep -f "tailscaled --tun=userspace-networking" >/dev/null; then
  echo "  이미 실행 중"
else
  [ -x "$TSD" ] || { echo "  $TSD 가 없다. 정적 바이너리를 ~/bin 에 설치할 것" >&2; exit 1; }
  mkdir -p "$TS_DIR"
  # TUN 이 없는 컨테이너라 userspace 모드다. root 불필요.
  # 대가: 이 노드는 자기 tailnet IP 로 자기한테 접속할 수 없다 (자기 참조는 localhost 사용)
  nohup "$TSD" --tun=userspace-networking --socket="$TS_SOCK" --statedir="$TS_DIR" \
    > "$TS_DIR/daemon.log" 2>&1 &
  echo "  기동 (PID $!) → $TS_DIR/daemon.log"
  for _ in $(seq 1 30); do
    [ -S "$TS_SOCK" ] && break
    sleep 1
  done
fi

echo "== 2/3 tailnet 로그인 확인 =="
if ts status >/dev/null 2>&1 && ts ip -4 >/dev/null 2>&1; then
  TS_IP="$(ts ip -4 | head -1)"
  echo "  로그인됨 — tailnet IP $TS_IP"
else
  echo "  로그인 안 됨. auth key 로 한 번 붙여야 한다:" >&2
  echo "    $TS --socket=$TS_SOCK up --authkey=tskey-... --hostname=school-gpu" >&2
  exit 1
fi

echo "== 3/3 워커 =="
VOICES_FILE="${VOICES_FILE:-$HOME/bada-qwen3-tts/voices.json}" \
QWEN_PYTHON="${QWEN_PYTHON:-$HOME/bada-qwen3-tts/fork/.venv/bin/python}" \
QWEN_SERVER_DIR="${QWEN_SERVER_DIR:-$HOME/bada-qwen3-tts/fork}" \
CC="${CC:-$HOME/gcc-env/bin/x86_64-conda-linux-gnu-gcc}" \
CXX="${CXX:-$HOME/gcc-env/bin/x86_64-conda-linux-gnu-g++}" \
  bash "$ROOT/launch_workers.sh" "${1:-1,2}" "${2:-2}" "${3:-8010}"

echo
echo "EC2 가 쓸 값 (tailnet IP 기준):"
PORTS="${2:-2}"
GPUS="${1:-1,2}"
BASE="${3:-8010}"
n=0
for _ in ${GPUS//,/ }; do n=$((n + PORTS)); done
urls=""
for i in $(seq 0 $((n - 1))); do urls="${urls:+$urls,}http://$TS_IP:$((BASE + i))"; done
echo "  QWEN_TTS_URLS=$urls"
