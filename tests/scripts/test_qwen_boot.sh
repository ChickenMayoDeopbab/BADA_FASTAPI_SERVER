#!/usr/bin/env bash
# scripts/qwen_tts_server/{boot,launch_workers}.sh 재시작 복구 검증 (plan 0058)
# 학교 GPU 서버 컨테이너가 재시작된 직후 상태 — /home 에 남은 옛 소켓·PID 파일·로그, 30계정이 공유하는
# 컨테이너의 남의 프로세스 — 를 임시 HOME 과 PATH 스텁(tailscale/tailscaled/python/id/date)으로 재현한다.
# GPU·tailscale 없이 맥에서 돈다.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BOOT="$ROOT/scripts/qwen_tts_server/boot.sh"
LAUNCH="$ROOT/scripts/qwen_tts_server/launch_workers.sh"
WORK=$(mktemp -d -t qwen-boot-test.XXXXXX)
ME=$(id -u)
OTHER_UID=$((ME + 1))
STAMP=20261003-065300  # date 스텁이 찍는 시각 — 옮긴 로그 이름을 예측하려고 고정

cleanup() { pkill -u "$ME" -f "$WORK" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

PASS=0
ok()   { PASS=$((PASS + 1)); echo "  ok: $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 0. 문법 / (있으면) shellcheck ------------------------------------------------
for s in "$BOOT" "$LAUNCH"; do
  bash -n "$s" || fail "bash -n $s"
done
ok "bash -n 문법 통과 (boot.sh, launch_workers.sh)"

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$BOOT" "$LAUNCH" || fail "shellcheck"
  ok "shellcheck 통과"
fi

# --- 스텁 ------------------------------------------------------------------------
STUB="$WORK/stub"
mkdir -p "$STUB/bin" "$STUB/fakeid"

# 오래 사는 프로세스. 명령줄(인자)은 그대로 남아 ps/pgrep 에 보인다. TERM 을 받으면 자식까지 정리한다
cat >"$STUB/idle" <<'EOF'
#!/usr/bin/env bash
trap 'kill $! 2>/dev/null; exit 0' TERM
sleep 300 &
wait
EOF

# 가짜 워커(python). 기동 기록만 남기고 오래 산다 — uvicorn 이 포트를 잡는 대신
cat >"$STUB/python" <<'EOF'
#!/usr/bin/env bash
echo "worker start $*" >>"$STUB_LOG"
echo "worker log $$"
trap 'kill $! 2>/dev/null; exit 0' TERM
sleep 300 &
wait
EOF

# 가짜 tailscaled: 소켓을 바로 만들고(옛 소켓은 지우고 다시 bind), STUB_TSD_DELAY 초 뒤에야
# tailnet 에 붙은 것으로 표시한다(<소켓>.ready). STUB_NEVER_READY=1 이면 끝내 안 붙는다(로그인 안 됨)
cat >"$STUB/tailscaled" <<'EOF'
#!/usr/bin/env bash
sock=""
for a in "$@"; do case "$a" in --socket=*) sock="${a#--socket=}" ;; esac; done
echo "tailscaled start $sock" >>"$STUB_LOG"
echo "daemon log $$"
rm -f "$sock.ready"
( cd "$(dirname "$sock")" && rm -f "$(basename "$sock")" && \
  python3 -c 'import socket, sys; socket.socket(socket.AF_UNIX).bind(sys.argv[1])' "$(basename "$sock")" )
trap 'kill $! 2>/dev/null; exit 0' TERM
if [ "${STUB_NEVER_READY:-0}" != 1 ]; then
  sleep "${STUB_TSD_DELAY:-0}" &
  wait
  touch "$sock.ready"
fi
sleep 300 &
wait
EOF

# 가짜 tailscale CLI: 데몬이 붙었을 때(<소켓>.ready)만 status/ip 가 성공한다
cat >"$STUB/tailscale" <<'EOF'
#!/usr/bin/env bash
sock="${1#--socket=}"
shift
[ -e "$sock.ready" ] || { echo "stub: tailscaled 응답 없음 ($sock)" >&2; exit 1; }
case "$1" in
  status) echo "100.100.1.1  school-gpu  stub  linux  -" ;;
  ip)     echo "100.100.1.1" ;;
  *)      exit 1 ;;
esac
EOF

cat >"$STUB/bin/date" <<EOF
#!/usr/bin/env bash
echo "$STAMP"
EOF

# 남의 프로세스는 sudo 없이 만들 수 없다 — 반대로 스크립트가 자기 UID 를 다르게 알게 해서
# "내 프로세스 = 남의 것" 을 만든다
cat >"$STUB/fakeid/id" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "-u" ]; then echo "$STUB_UID"; else exec /usr/bin/id "$@"; fi
EOF
chmod +x "$STUB/idle" "$STUB/python" "$STUB/tailscaled" "$STUB/tailscale" "$STUB/bin/date" "$STUB/fakeid/id"

# 케이스 하나 = HOME 하나. boot.sh·launch_workers.sh 의 기본 경로가 전부 그 아래로 간다
new_case() {
  local c="$WORK/$1" h
  h="$c/home"
  mkdir -p "$h/bin" "$h/.tailscale" "$h/bada-qwen3-tts/fork/.venv/bin"
  cp "$STUB/tailscale" "$STUB/tailscaled" "$h/bin/"
  cp "$STUB/python" "$h/bada-qwen3-tts/fork/.venv/bin/python"
  echo '{}' >"$h/bada-qwen3-tts/voices.json"
  : >"$h/bada-qwen3-tts/fork/server.py"
  : >"$c/stub.log"
  echo "$c"
}

end_case() { pkill -u "$ME" -f "$1/" 2>/dev/null || true; }

# in_case <case-dir> [VAR=값 ...] <명령...> — 케이스 HOME·스텁 환경에서 명령을 돌린다 (개발자 환경 값은 지운다)
in_case() {
  local c=$1
  shift
  env -u VOICES_FILE -u QWEN_PYTHON -u QWEN_SERVER_DIR -u QWEN_RUN_DIR -u CC -u CXX -u TS_WAIT \
    HOME="$c/home" PATH="$STUB/bin:$PATH" STUB_LOG="$c/stub.log" TS_WAIT=8 \
    VOICES_FILE="$c/home/bada-qwen3-tts/voices.json" \
    QWEN_PYTHON="$c/home/bada-qwen3-tts/fork/.venv/bin/python" \
    QWEN_SERVER_DIR="$c/home/bada-qwen3-tts/fork" CC=cc CXX=c++ "$@"
}

# run <boot|launch> <case-dir> [VAR=값 ...] -- [스크립트 인자...]
run() {
  local which=$1 c=$2 script
  shift 2
  [ "$which" = boot ] && script="$BOOT" || script="$LAUNCH"
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  [ "${1:-}" = "--" ] && shift
  in_case "$c" ${envs[@]+"${envs[@]}"} bash "$script" "$@" >>"$c/out.log" 2>&1
}

pid_dir() { echo "$1/home/bada-qwen3-tts/run/pid"; }
log_dir() { echo "$1/home/bada-qwen3-tts/run/log"; }
args_of() { ps -ww -o args= -p "$1" 2>/dev/null || true; }
# kill -0 은 끝났지만 거둬지지 않은 좀비도 살아 있다고 답한다. 학교 컨테이너는 PID 1 이 jupyterhub(파이썬)라
# 고아 프로세스를 거두지 않아, stop 으로 내린 워커가 좀비(상태 Z)로 남는다 — 상태로 판단한다
alive() {
  local s
  s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
  case "${s// /}" in ''|Z*) return 1 ;; esac
}
count()   { grep -c "$1" "$2" || true; }

# 워커 기동 직후는 fork→exec 사이라 명령줄이 아직 바뀌는 중일 수 있다 — 붙을 때까지 잠깐 기다린다
wait_args() {  # wait_args <pid> <부분 문자열>
  for _ in $(seq 1 200); do
    case "$(args_of "$1")" in *"$2"*) return 0 ;; esac
    sleep 0.05
  done
  return 1
}

# 워커는 백그라운드로 뜨므로 스크립트가 끝난 직후엔 기록이 아직 없을 수 있다
wait_grep() {  # wait_grep <패턴> <파일>
  for _ in $(seq 1 200); do
    grep -q "$1" "$2" 2>/dev/null && return 0
    sleep 0.05
  done
  return 1
}

# 남의/엉뚱한 프로세스를 띄워 PID 를 돌려준다
spawn() {  # spawn <case-dir> <이름> [인자...]
  local c=$1 n=$2
  shift 2
  cp "$STUB/idle" "$c/$n"
  "$c/$n" "$@" >/dev/null 2>&1 &
  echo $!
}

# 옛 소켓: bind 하고 닫으면 파일만 남는다 (재시작 전 데몬이 남긴 것)
stale_socket() {
  ( cd "$(dirname "$1")" && python3 -c 'import socket, sys; socket.socket(socket.AF_UNIX).bind(sys.argv[1])' "$(basename "$1")" )
  [ -S "$1" ] || fail "시험 준비: 옛 소켓을 못 만듦 ($1)"
}

# --- 함정 1: 남의 tailscaled 를 우리 것으로 오판 ----------------------------------
# 1a. 같은 명령줄이지만 다른 소켓(= 다른 사람 홈)의 데몬
C=$(new_case t1a)
mkdir -p "$C/other/.tailscale"
OTHER_TSD=$(spawn "$C" tailscaled --tun=userspace-networking \
  --socket="$C/other/.tailscale/tailscaled.sock" --statedir="$C/other/.tailscale")
wait_args "$OTHER_TSD" "--tun=userspace-networking" || fail "1a: 남의 데몬 흉내가 안 뜸"
run boot "$C" STUB_TSD_DELAY=1 -- 1 1 8010 || fail "1a: 남의 데몬이 있다고 복구가 실패함 ($(cat "$C/out.log"))"
grep -q "이미 실행 중" "$C/out.log" && fail "1a: 다른 소켓의 데몬을 우리 것으로 봄"
[ "$(count "tailscaled start $C/home/.tailscale/tailscaled.sock" "$C/stub.log")" = 1 ] \
  || fail "1a: 우리 데몬을 띄우지 않음 ($(cat "$C/stub.log"))"
wait_grep "worker start" "$C/stub.log" || fail "1a: 워커까지 가지 못함"
ok "남의 tailscaled(다른 소켓): 건너뛰지 않고 우리 데몬 기동 → 워커까지"
end_case "$C"

# 1b. 우리 소켓 경로지만 다른 UID 의 데몬 (pgrep -u 가 걸러야 한다)
C=$(new_case t1b)
SOCK="$C/home/.tailscale/tailscaled.sock"
STUB_LOG="$C/stub.log" "$C/home/bin/tailscaled" --tun=userspace-networking --socket="$SOCK" \
  --statedir="$C/home/.tailscale" >/dev/null 2>&1 &
for _ in $(seq 1 200); do [ -e "$SOCK.ready" ] && break; sleep 0.05; done
run boot "$C" PATH="$STUB/fakeid:$STUB/bin:$PATH" STUB_UID="$OTHER_UID" -- 1 1 8010 \
  || fail "1b: 실패함 ($(cat "$C/out.log"))"
grep -q "이미 실행 중" "$C/out.log" && fail "1b: 다른 UID 의 데몬을 우리 것으로 봄"
[ "$(count "tailscaled start" "$C/stub.log")" = 2 ] || fail "1b: 우리 데몬을 띄우지 않음 ($(cat "$C/stub.log"))"
ok "다른 UID 의 tailscaled: 건너뛰지 않고 우리 데몬 기동"
end_case "$C"

# 1c. 진짜 우리 데몬(내 UID·우리 소켓)이 떠 있으면 다시 띄우지 않는다
C=$(new_case t1c)
SOCK="$C/home/.tailscale/tailscaled.sock"
STUB_LOG="$C/stub.log" "$C/home/bin/tailscaled" --tun=userspace-networking --socket="$SOCK" \
  --statedir="$C/home/.tailscale" >/dev/null 2>&1 &
for _ in $(seq 1 200); do [ -e "$SOCK.ready" ] && break; sleep 0.05; done
run boot "$C" -- 1 1 8010 || fail "1c: 실패함 ($(cat "$C/out.log"))"
grep -q "이미 실행 중" "$C/out.log" || fail "1c: 우리 데몬을 못 알아봄"
[ "$(count "tailscaled start" "$C/stub.log")" = 1 ] || fail "1c: 데몬을 또 띄움"
grep -q "로그인됨" "$C/out.log" || fail "1c: 로그인 확인 실패"
ok "우리 tailscaled 가 떠 있으면: 다시 띄우지 않고 진행"
end_case "$C"

# 1d. 홈 경로에 정규식 글자(+)가 있어도 우리 데몬을 알아본다 — 소켓 경로를 pgrep 패턴에 넣으면
#     "d+x" 가 "dx"·"ddx" 로 해석돼 글자 그대로의 경로와 안 맞는다
C=$(new_case "t1d+x")
SOCK="$C/home/.tailscale/tailscaled.sock"
STUB_LOG="$C/stub.log" "$C/home/bin/tailscaled" --tun=userspace-networking --socket="$SOCK" \
  --statedir="$C/home/.tailscale" >/dev/null 2>&1 &
for _ in $(seq 1 200); do [ -e "$SOCK.ready" ] && break; sleep 0.05; done
run boot "$C" -- 1 1 8010 || fail "1d: 실패함 ($(cat "$C/out.log"))"
grep -q "이미 실행 중" "$C/out.log" || fail "1d: 홈 경로의 + 때문에 우리 데몬을 못 알아봄"
[ "$(count "tailscaled start" "$C/stub.log")" = 1 ] || fail "1d: 데몬을 또 띄움"
ok "홈 경로에 정규식 글자(+): 우리 tailscaled 를 글자 그대로 알아봄"
pkill -u "$ME" -f "$WORK/t1d" 2>/dev/null || true  # end_case 의 패턴도 정규식이라 + 앞까지만 준다

# --- 함정 2: 옛 소켓 파일 때문에 대기가 바로 끝나 "로그인 안 됨" 거짓 실패 -----------
# 2a. 옛 소켓이 남아 있고 데몬은 2초 뒤에야 tailnet 에 붙는다 + 함정 4: daemon.log 보존
C=$(new_case t2a)
SOCK="$C/home/.tailscale/tailscaled.sock"
stale_socket "$SOCK"
echo "OLD-DAEMON-LOG before restart" >"$C/home/.tailscale/daemon.log"
run boot "$C" STUB_TSD_DELAY=2 -- 1 1 8010 \
  || fail "2a: 옛 소켓 때문에 한 번에 복구되지 않음 ($(cat "$C/out.log"))"
grep -q "로그인됨 — tailnet IP 100.100.1.1" "$C/out.log" || fail "2a: 로그인 확인 실패"
grep -q "QWEN_TTS_URLS=http://100.100.1.1:8010" "$C/out.log" || fail "2a: EC2 용 URL 이 안 나옴"
wait_grep "worker start" "$C/stub.log" || fail "2a: 워커까지 가지 못함"
ok "옛 소켓 + 2초 늦은 데몬: 기다렸다가 한 번에 워커까지"
grep -q "OLD-DAEMON-LOG" "$C/home/.tailscale/daemon.log.$STAMP" 2>/dev/null \
  || fail "4: 이전 daemon.log 가 보존되지 않음 ($(ls "$C/home/.tailscale"))"
grep -q "OLD-DAEMON-LOG" "$C/home/.tailscale/daemon.log" && fail "4: 새 daemon.log 에 옛 내용이 섞임"
grep -q "daemon log" "$C/home/.tailscale/daemon.log" || fail "4: 새 daemon.log 가 안 쓰임"
ok "daemon.log: 이전 내용은 daemon.log.$STAMP 로, 새 데몬은 새 파일에"
end_case "$C"

# 2b. 끝내 로그인이 안 되면 TS_WAIT 안에 "로그인 안 됨" 으로 끝나고 워커는 띄우지 않는다
C=$(new_case t2b)
stale_socket "$C/home/.tailscale/tailscaled.sock"
start=$(date +%s)
if run boot "$C" STUB_NEVER_READY=1 TS_WAIT=2 -- 1 1 8010; then
  fail "2b: 로그인이 안 됐는데 0 으로 끝남"
fi
elapsed=$(( $(date +%s) - start ))
grep -q "로그인 안 됨" "$C/out.log" || fail "2b: 로그인 안내가 없음 ($(cat "$C/out.log"))"
grep -q "worker start" "$C/stub.log" && fail "2b: 로그인 안 됐는데 워커를 띄움"
[ "$elapsed" -le 10 ] || fail "2b: TS_WAIT=2 인데 ${elapsed}초 걸림"
ok "로그인 안 됨: TS_WAIT 안에 안내하고 끝, 워커 안 띄움 (${elapsed}초)"
end_case "$C"

# --- 함정 3·5: 옛 PID 파일의 번호가 다른 프로세스에 재사용됨 --------------------------
# 3a. PID 파일이 내 다른 프로세스를 가리킴 → 건너뛰지 않고 띄운다. PID 파일은 워커 자신을 가리킨다(함정 5)
C=$(new_case t3a)
P=$(pid_dir "$C"); mkdir -p "$P"
REUSED=$(spawn "$C" jupyter-kernel --ip=127.0.0.1)
echo "$REUSED" >"$P/gpu1-w1-p8010.pid"
run launch "$C" -- 1 1 8010 || fail "3a: 실패함 ($(cat "$C/out.log"))"
grep -q "이미 떠 있음" "$C/out.log" && fail "3a: 재사용된 PID 를 워커로 봄"
NEW=$(cat "$P/gpu1-w1-p8010.pid")
[ "$NEW" != "$REUSED" ] || fail "3a: PID 파일이 그대로"
wait_args "$NEW" "uvicorn server:app --host 127.0.0.1 --port 8010" \
  || fail "5: PID 파일이 워커가 아닌 프로세스를 가리킴 ($(args_of "$NEW"))"
alive "$REUSED" || fail "3a: 기동하면서 엉뚱한 프로세스를 죽임"
ok "PID 재사용(내 다른 프로세스): 워커 새로 기동, PID 파일 = 워커 자신"
end_case "$C"

# 3b. stop 이 재사용된 PID 를 kill 하지 않는다
C=$(new_case t3b)
P=$(pid_dir "$C"); mkdir -p "$P"
REUSED=$(spawn "$C" jupyter-kernel --ip=127.0.0.1)
echo "$REUSED" >"$P/gpu1-w1-p8010.pid"
run launch "$C" -- stop || fail "3b: stop 실패 ($(cat "$C/out.log"))"
sleep 0.2
alive "$REUSED" || fail "3b: stop 이 워커가 아닌 프로세스를 kill 함"
[ -e "$P/gpu1-w1-p8010.pid" ] && fail "3b: PID 파일이 안 지워짐"
ok "stop: 재사용된 PID 는 kill 하지 않고 PID 파일만 정리"
end_case "$C"

# 3c. 진짜 워커면 건너뛰고(로그도 그대로), stop 은 워커를 남김없이 내린다(함정 5)
C=$(new_case t3c)
P=$(pid_dir "$C"); L=$(log_dir "$C")
run launch "$C" -- 1 1 8010 || fail "3c: 첫 기동 실패 ($(cat "$C/out.log"))"
W=$(cat "$P/gpu1-w1-p8010.pid")
wait_args "$W" "--port 8010" || fail "5: PID 파일이 워커가 아닌 프로세스를 가리킴 ($(args_of "$W"))"
wait_grep "worker start" "$C/stub.log" || fail "3c: 워커 기동 기록이 없음"
run launch "$C" -- 1 1 8010 || fail "3c: 두 번째 실행 실패"
grep -q "이미 떠 있음 gpu1-w1-p8010" "$C/out.log" || fail "3c: 살아 있는 워커를 못 알아봄 ($(cat "$C/out.log"))"
[ "$(count "worker start" "$C/stub.log")" = 1 ] || fail "3c: 워커를 또 띄움"
[ "$(cat "$P/gpu1-w1-p8010.pid")" = "$W" ] || fail "3c: PID 파일이 바뀜"
ls "$L" | grep -q "gpu1-w1-p8010.log." && fail "3c: 건너뛴 워커의 로그를 옮김"
run launch "$C" -- stop || fail "3c: stop 실패"
for _ in $(seq 1 200); do alive "$W" || break; sleep 0.05; done
alive "$W" && fail "3c: stop 뒤에도 워커가 살아 있음"
sleep 0.2
pgrep -u "$ME" -f "$C/home/bada-qwen3-tts/fork/.venv/bin/python" >/dev/null \
  && fail "5: stop 뒤에도 워커 프로세스가 남음 (PID 파일이 워커가 아니었다)"
ok "살아 있는 워커: 건너뛰고 로그 유지, stop 은 워커를 남김없이 내림"
end_case "$C"

# 3d. 워커처럼 생겼지만 다른 포트 → 이 포트의 워커가 아니다
C=$(new_case t3d)
P=$(pid_dir "$C"); mkdir -p "$P"
OTHERPORT=$(spawn "$C" python -m uvicorn server:app --host 127.0.0.1 --port 8011)
wait_args "$OTHERPORT" "--port 8011" || fail "3d: 준비 실패"
echo "$OTHERPORT" >"$P/gpu1-w1-p8010.pid"
run launch "$C" -- 1 1 8010 || fail "3d: 실패함"
grep -q "기동 gpu1-w1-p8010" "$C/out.log" || fail "3d: 다른 포트의 워커를 이 포트 워커로 봄 ($(cat "$C/out.log"))"
ok "다른 포트의 워커를 가리키는 PID 파일: 새로 기동"
end_case "$C"

# 3e. 워커처럼 생겼지만 다른 UID → 내 워커가 아니다
C=$(new_case t3e)
P=$(pid_dir "$C"); mkdir -p "$P"
LOOKALIKE=$(spawn "$C" python -m uvicorn server:app --host 127.0.0.1 --port 8010)
wait_args "$LOOKALIKE" "--port 8010" || fail "3e: 준비 실패"
echo "$LOOKALIKE" >"$P/gpu1-w1-p8010.pid"
run launch "$C" PATH="$STUB/fakeid:$STUB/bin:$PATH" STUB_UID="$OTHER_UID" -- 1 1 8010 || fail "3e: 실패함"
grep -q "기동 gpu1-w1-p8010" "$C/out.log" || fail "3e: 다른 UID 의 워커를 내 워커로 봄 ($(cat "$C/out.log"))"
run launch "$C" PATH="$STUB/fakeid:$STUB/bin:$PATH" STUB_UID="$OTHER_UID" -- stop || fail "3e: stop 실패"
ok "다른 UID 의 워커를 가리키는 PID 파일: 새로 기동"
end_case "$C"

# 3f. 죽은 PID · 쓰레기 · 빈 PID 파일 → 오류 없이 새로 기동
C=$(new_case t3f)
P=$(pid_dir "$C"); mkdir -p "$P"
sleep 0 & DEAD=$!; wait "$DEAD" || true
echo "$DEAD" >"$P/gpu1-w1-p8010.pid"
echo "abc" >"$P/gpu1-w2-p8011.pid"
: >"$P/gpu2-w1-p8012.pid"
run launch "$C" -- 1,2 2 8010 || fail "3f: 실패함 ($(cat "$C/out.log"))"
[ "$(count "기동 gpu" "$C/out.log")" = 4 ] || fail "3f: 4개 다 기동하지 않음 ($(cat "$C/out.log"))"
run launch "$C" -- stop || fail "3f: stop 실패"
ok "죽은 PID·쓰레기·빈 PID 파일: 오류 없이 4개 기동"
end_case "$C"

# 3g. 끝났지만 거둬지지 않은 워커(좀비)는 워커가 아니다. 학교 컨테이너는 PID 1(jupyterhub)이 고아를 거두지 않아
#     내리거나 죽은 워커가 좀비로 남는다 — 번호만 보면(kill -0) 살아 있다고 보고 다시 띄우지 않는다
C=$(new_case t3g)
P=$(pid_dir "$C")
# 런처를 source 한 셸이 끝에 exec sleep 으로 바뀌어, 워커의 부모가 거두지 않는 프로세스가 된다 (= 서버의 PID 1)
in_case "$C" bash -c 'echo $$ >"$2"; . "$1" 1 1 8010 >/dev/null 2>&1; exec sleep 120' _ "$LAUNCH" "$C/holder.pid" &
for _ in $(seq 1 200); do [ -s "$P/gpu1-w1-p8010.pid" ] && break; sleep 0.05; done
Z=$(cat "$P/gpu1-w1-p8010.pid" 2>/dev/null) || fail "3g: 준비 실패 — 워커가 안 뜸"
wait_args "$Z" "--port 8010" || fail "3g: 준비 실패 ($(args_of "$Z"))"
run launch "$C" -- stop || fail "3g: stop 실패"
for _ in $(seq 1 200); do case "$(ps -o stat= -p "$Z" 2>/dev/null)" in *Z*) break ;; esac; sleep 0.05; done
kill -0 "$Z" 2>/dev/null || fail "3g: 준비 실패 — 좀비가 안 생김 (이 시험의 전제)"
alive "$Z" && fail "3g: alive() 가 좀비를 살아 있다고 봄"
echo "$Z" >"$P/gpu1-w1-p8010.pid"  # stop 이 지운 PID 파일을 되살린다 — 재시작 직후 옛 파일이 남은 것처럼
run launch "$C" -- 1 1 8010 || fail "3g: 실패함 ($(cat "$C/out.log"))"
grep -q "이미 떠 있음" "$C/out.log" && fail "3g: 좀비를 살아 있는 워커로 보고 기동을 건너뜀"
[ "$(cat "$P/gpu1-w1-p8010.pid")" != "$Z" ] || fail "3g: PID 파일이 좀비를 그대로 가리킴"
ok "끝난 워커가 좀비로 남아도(PID 1 이 안 거둠): 워커 아님으로 보고 새로 기동"
run launch "$C" -- stop || true
kill "$(cat "$C/holder.pid")" 2>/dev/null || true
end_case "$C"

# --- 함정 4: 워커 로그 덮어쓰기 --------------------------------------------------
C=$(new_case t4)
L=$(log_dir "$C"); mkdir -p "$L"
echo "OLD-WORKER-LOG crash trace" >"$L/gpu1-w1-p8010.log"
echo "OLDER backup" >"$L/gpu1-w1-p8010.log.$STAMP"
run launch "$C" -- 1 1 8010 || fail "4: 실패함 ($(cat "$C/out.log"))"
[ "$(cat "$L/gpu1-w1-p8010.log.$STAMP")" = "OLDER backup" ] || fail "4: 같은 이름의 이전 백업을 덮어씀"
BACKUP=$(ls "$L" | grep "^gpu1-w1-p8010.log.$STAMP\.[0-9]*$" || true)
[ -n "$BACKUP" ] && grep -q "OLD-WORKER-LOG" "$L/$BACKUP" || fail "4: 이전 워커 로그가 보존되지 않음 ($(ls "$L"))"
wait_grep "worker log" "$L/gpu1-w1-p8010.log" || true
grep -q "OLD-WORKER-LOG" "$L/gpu1-w1-p8010.log" && fail "4: 새 로그에 옛 내용이 섞임"
grep -q "worker log" "$L/gpu1-w1-p8010.log" || fail "4: 새 워커 로그가 안 쓰임"
run launch "$C" -- stop || true
ok "워커 로그: 이전 내용은 시각 붙은 파일로, 같은 이름이 있으면 덮어쓰지 않음"
end_case "$C"

echo "PASS: $PASS/$PASS 시나리오 통과"
