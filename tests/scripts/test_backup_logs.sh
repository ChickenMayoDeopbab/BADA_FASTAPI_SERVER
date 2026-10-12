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
    if [[ "${STUB_TOKEN:-0}" == "1" ]]; then
      echo "2026-10-12 20:25:18.945 [INFO] uvicorn.error: ('10.0.0.1', 51234) - \"WebSocket /ws/voice/s1?lang=ko&token=$STUB_JWT\" [accepted]"
      echo "2026-10-12 20:25:18.964 [INFO] uvicorn.access: 10.0.0.1:51235 - \"GET /ws/voice/s1?TOKEN=opaque-123&x=1 HTTP/1.1\" 404"
      echo "Authorization: Bearer $STUB_JWT" >&2
      printf 'broken utf-8 \xff\xfe bytes\n'
      echo "{\"message\": \"('10.0.0.1', 51236) - \\\"WebSocket /ws/voice/s1?token=$STUB_JWT\\\" [accepted]\"}"
      echo "two ?token=first-secret&token=second-secret and $STUB_JWT $STUB_JWT unsigned eyJhbGciOiJub25lIn0.eyJzdWIiOiI3In0."
    fi
    if [[ "${STUB_DOCKER_FAIL:-0}" == "1" ]]; then
      exit 1
    fi
    ;;
esac
EOF

cat >"$STUB_BIN/aws" <<'EOF'
#!/usr/bin/env bash
echo "aws $* cred:${AWS_ACCESS_KEY_ID:-none}" >>"$STUB_LOG"
# aws s3 cp <src> <dst> — 올라가는 파일 내용을 검사할 수 있게 복사해 둔다
if [[ -n "${STUB_UPLOAD:-}" && "$1 $2" == "s3 cp" ]]; then
  cp "$3" "$STUB_UPLOAD"
fi
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

# --- 7. 토큰 가림: S3 에 올라가는 로그에 JWT·쿼리 token 값이 없다 (계획 0060) ----------
# 첫 실행(--until 만)과 증분 실행(--since, 운영 크론·배포 전 백업의 보통 경로) 두 분기 모두.
# UTF-8 로케일에서 깨진 바이트가 섞여도 가림 단계가 실패하면 안 된다(BSD sed 는 RE 오류로 멈춘다).
C7="$WORK/case7"; mkdir -p "$C7"
JWT="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI3IiwidHlwZSI6IkFDQ0VTUyJ9.c2lnbmF0dXJlLXZhbHVl"
for run in first incremental; do
  : >"$C7/stub.log"
  if [[ "$run" == incremental ]]; then echo "$T1" >"$C7/state"; fi
  run_backup "$C7" LC_ALL=en_US.UTF-8 S3_BUCKET=test-bucket STUB_TOKEN=1 STUB_JWT="$JWT" STUB_UPLOAD="$C7/upload-$run.gz" \
    || fail "case7($run): 스크립트가 실패함 ($(cat "$C7/out.log"))"
  if [[ "$run" == incremental ]]; then
    grep -q -- "--since $T1" "$C7/stub.log" || fail "case7($run): 증분 분기를 안 탐"
  else
    grep -q -- "--since" "$C7/stub.log" && fail "case7($run): 첫 실행인데 --since 가 붙음"
  fi
  UPLOADED=$(gzip -dc "$C7/upload-$run.gz")
  [[ "$UPLOADED" != *"$JWT"* ]] || fail "case7($run): 올라간 로그에 JWT 가 남음"
  [[ "$UPLOADED" != *"opaque-123"* ]] || fail "case7($run): 올라간 로그에 쿼리 token 값이 남음"
  [[ "$UPLOADED" == *'"WebSocket /ws/voice/s1?lang=ko&token=[REDACTED]" [accepted]'* ]] \
    || fail "case7($run): WebSocket 접속 줄이 가려진 모양이 아님 ($UPLOADED)"
  [[ "$UPLOADED" == *'"GET /ws/voice/s1?TOKEN=[REDACTED]&x=1 HTTP/1.1" 404'* ]] \
    || fail "case7($run): 접근 로그 줄이 가려진 모양이 아님 ($UPLOADED)"
  [[ "$UPLOADED" == *"Authorization: Bearer [REDACTED]"* ]] || fail "case7($run): stderr 의 JWT 가 안 가려짐 ($UPLOADED)"
  [[ "$UPLOADED" == *'"message": "stub log line"'* ]] || fail "case7($run): 토큰 없는 줄이 바뀜 ($UPLOADED)"
  [[ "$UPLOADED" == *"broken utf-8 "*" bytes"* ]] || fail "case7($run): 깨진 바이트 줄이 빠짐"
  [[ "$UPLOADED" == *'"WebSocket /ws/voice/s1?token=[REDACTED]\" [accepted]"}'* ]] \
    || fail "case7($run): JSON 줄의 \\\" 이스케이프가 깨짐 ($UPLOADED)"
  [[ "$UPLOADED" != *"-secret"* && "$UPLOADED" != *"eyJzdWIiOiI3In0."* ]] \
    || fail "case7($run): 한 줄의 두 번째 비밀이나 서명 없는 JWT 가 남음 ($UPLOADED)"
  [[ "$UPLOADED" == *"two ?token=[REDACTED]&token=[REDACTED] and [REDACTED] [REDACTED] unsigned [REDACTED]"* ]] \
    || fail "case7($run): 여러 비밀 줄이 가려진 모양이 아님 ($UPLOADED)"
done
ok "토큰 가림: 첫 실행·증분 모두 쿼리 token 값·JWT 를 [REDACTED] 로, 나머지 줄은 그대로 업로드"

# --- 8. docker logs 실패: 비정상 종료 + 업로드 없음 + 상태 유지 ------------------------
# 가림 단계를 파이프로 붙여서 docker logs 실패를 잡는 건 이제 pipefail 뿐이다.
C8="$WORK/case8"; mkdir -p "$C8"; : >"$C8/stub.log"
echo "$T1" >"$C8/state"
if run_backup "$C8" S3_BUCKET=test-bucket STUB_DOCKER_FAIL=1; then
  fail "case8: docker logs 가 실패했는데 0 으로 종료"
fi
grep -q "aws s3 cp" "$C8/stub.log" && fail "case8: docker logs 가 실패했는데 업로드함"
[[ "$(cat "$C8/state")" == "$T1" ]] || fail "case8: docker logs 가 실패했는데 상태가 갱신됨"
ok "docker logs 실패: 종료코드≠0 + 업로드 없음 + 상태 유지"

echo "PASS: $PASS/$PASS 시나리오 통과"
