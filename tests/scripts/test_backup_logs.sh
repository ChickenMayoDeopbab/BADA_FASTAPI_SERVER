#!/usr/bin/env bash
# scripts/backup_logs.sh 동작 검증 (plan 0025 / F36)
# docker/aws 를 PATH 스텁으로 대체해 실제 도커/S3 없이 시나리오를 검증한다.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$ROOT/scripts/backup_logs.sh"
WORK=$(mktemp -d -t backup-logs-test.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

PASS=0
ok()   { PASS=$((PASS + 1)); echo "  ok: $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 0. 문법 / (있으면) shellcheck / compose config ----------------------------
bash -n "$SCRIPT" || fail "bash -n"
ok "bash -n 문법 통과"

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$SCRIPT" || fail "shellcheck"
  ok "shellcheck 통과"
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  IMAGE=ghcr.io/test/app:test docker compose \
    -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.prod.yml" config -q \
    || fail "docker compose config"
  ok "compose config (prod 오버레이 + 로테이션) 유효"
else
  echo "  skip: docker 없음 — compose config 검사 생략"
fi

# --- 스텁 준비 -------------------------------------------------------------------
STUB_BIN="$WORK/bin"
mkdir -p "$STUB_BIN"

cat >"$STUB_BIN/docker" <<'EOF'
#!/usr/bin/env bash
echo "docker $*" >>"$STUB_LOG"
case "$1" in
  compose)
    # docker compose ps -q <service>
    echo "stub-container-id"
    ;;
  logs)
    if [[ "${STUB_EMPTY:-0}" == "1" ]]; then
      exit 0
    fi
    echo '{"asctime": "2026-07-14T00:00:00", "levelname": "INFO", "message": "stub log line"}'
    ;;
esac
EOF

cat >"$STUB_BIN/aws" <<'EOF'
#!/usr/bin/env bash
echo "aws $* cred:${AWS_ACCESS_KEY_ID:-none}" >>"$STUB_LOG"
if [[ "${STUB_AWS_FAIL:-0}" == "1" ]]; then
  exit 1
fi
EOF
chmod +x "$STUB_BIN/docker" "$STUB_BIN/aws"

# 케이스 공통 실행기: run_backup <case-dir> [추가 env...]
run_backup() {
  local case_dir=$1
  shift
  env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_DEFAULT_REGION \
    PATH="$STUB_BIN:$PATH" \
    STUB_LOG="$case_dir/stub.log" \
    COMPOSE_DIR="$case_dir" \
    STATE_FILE="$case_dir/state" \
    "$@" bash "$SCRIPT" >"$case_dir/out.log" 2>&1
}

T1="2026-01-01T00:00:00Z"

# --- 1. 첫 실행: 전체 덤프 → 업로드 → 상태 파일 생성 ----------------------------
C1="$WORK/case1"; mkdir -p "$C1"; : >"$C1/stub.log"
run_backup "$C1" S3_BUCKET=test-bucket || fail "case1: 스크립트가 실패함 ($(cat "$C1/out.log"))"
grep -q "docker logs --until" "$C1/stub.log" || fail "case1: docker logs 호출 없음"
grep -q -- "--since" "$C1/stub.log" && fail "case1: 첫 실행인데 --since 가 붙음"
grep -Eq "aws s3 cp .* s3://test-bucket/logs/app/[0-9]{4}/[0-9]{2}/[0-9]{2}/app_container-start_.*\.log\.gz" "$C1/stub.log" \
  || fail "case1: S3 업로드 키 형식이 다름 ($(grep aws "$C1/stub.log"))"
[[ -s "$C1/state" ]] || fail "case1: 상태 파일이 안 만들어짐"
ok "첫 실행: 전체 덤프 + 날짜별 키 업로드 + 상태 기록"

# --- 2. 업로드 실패: 비정상 종료 + 상태 미갱신 (재시도 보장) ---------------------
C2="$WORK/case2"; mkdir -p "$C2"; : >"$C2/stub.log"
echo "$T1" >"$C2/state"
if run_backup "$C2" S3_BUCKET=test-bucket STUB_AWS_FAIL=1; then
  fail "case2: 업로드 실패인데 0 으로 종료"
fi
[[ "$(cat "$C2/state")" == "$T1" ]] || fail "case2: 실패했는데 상태가 갱신됨"
ok "업로드 실패: 종료코드≠0 + 상태 유지(다음 실행이 같은 구간 재시도)"

# --- 3. 증분 실행: --since <이전 상태> + 상태 전진 ------------------------------
C3="$WORK/case3"; mkdir -p "$C3"; : >"$C3/stub.log"
echo "$T1" >"$C3/state"
run_backup "$C3" S3_BUCKET=test-bucket || fail "case3: 스크립트가 실패함 ($(cat "$C3/out.log"))"
grep -q -- "--since $T1" "$C3/stub.log" || fail "case3: --since $T1 이 안 붙음"
grep -q "app_2026-01-01T00-00-00Z_" "$C3/stub.log" || fail "case3: 키에 증분 시작 시각이 없음"
[[ "$(cat "$C3/state")" != "$T1" ]] || fail "case3: 성공했는데 상태가 안 전진함"
ok "증분 실행: --since 이전 상태 + 성공 시 상태 전진"

# --- 4. 빈 증분: 업로드 생략 + 상태는 전진 --------------------------------------
C4="$WORK/case4"; mkdir -p "$C4"; : >"$C4/stub.log"
echo "$T1" >"$C4/state"
run_backup "$C4" S3_BUCKET=test-bucket STUB_EMPTY=1 || fail "case4: 스크립트가 실패함"
grep -q "aws s3 cp" "$C4/stub.log" && fail "case4: 빈 증분인데 업로드함"
[[ "$(cat "$C4/state")" != "$T1" ]] || fail "case4: 상태가 안 전진함"
ok "빈 증분: 업로드 생략 + 상태 전진"

# --- 5. .env 재사용: S3_BUCKET/AWS 자격증명 매핑 (+CRLF 개행 제거) ---------------
C5="$WORK/case5"; mkdir -p "$C5"; : >"$C5/stub.log"
# S3_BUCKET 줄은 일부러 CRLF — \r 이 값에 남으면 s3:// URL 매칭이 깨진다
printf 'S3_BUCKET=env-bucket\r\nAWS_ACCESS_KEY=env-key-123\nAWS_SECRET_KEY=env-secret-456\nAWS_REGION=ap-northeast-2\n' >"$C5/.env"
run_backup "$C5" || fail "case5: 스크립트가 실패함 ($(cat "$C5/out.log"))"
grep -q "s3://env-bucket/" "$C5/stub.log" || fail "case5: .env 의 S3_BUCKET 을 안 씀 (CRLF 미제거?)"
grep -q "cred:env-key-123" "$C5/stub.log" || fail "case5: AWS_ACCESS_KEY → AWS_ACCESS_KEY_ID 매핑 안 됨"
ok ".env 재사용: 버킷 + 자격증명 매핑 + CRLF 제거"

# --- 6. 서비스별 분리: SERVICE=db → 전용 prefix/파일명/기본 상태 파일 -------------
C6="$WORK/case6"; mkdir -p "$C6"; : >"$C6/stub.log"
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_DEFAULT_REGION \
  PATH="$STUB_BIN:$PATH" \
  STUB_LOG="$C6/stub.log" \
  COMPOSE_DIR="$C6" \
  HOME="$C6" \
  S3_BUCKET=test-bucket SERVICE=db \
  bash "$SCRIPT" >"$C6/out.log" 2>&1 || fail "case6: 스크립트가 실패함 ($(cat "$C6/out.log"))"
grep -q "compose ps -q db" "$C6/stub.log" || fail "case6: SERVICE=db 로 컨테이너를 안 찾음"
grep -Eq "s3://test-bucket/logs/db/[0-9]{4}/[0-9]{2}/[0-9]{2}/db_container-start_" "$C6/stub.log" \
  || fail "case6: 서비스별 prefix/파일명이 아님 ($(grep aws "$C6/stub.log"))"
[[ -s "$C6/.bada/last_log_backup_db" ]] || fail "case6: 기본 상태 파일에 서비스 접미사가 없음"
ok "서비스별 분리: prefix/파일명/기본 상태 파일에 SERVICE 반영"

echo "PASS: $PASS/$PASS 시나리오 통과"
