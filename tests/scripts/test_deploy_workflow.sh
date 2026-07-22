#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WF="$ROOT/.github/workflows/deploy.yml"

PASS=0
ok()   { PASS=$((PASS + 1)); echo "  ok: $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

[[ -f "$WF" ]] || fail "deploy.yml 없음"

if "$ROOT/.venv/bin/python" -c "import yaml" >/dev/null 2>&1; then
  "$ROOT/.venv/bin/python" -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" "$WF" \
    || fail "deploy.yml YAML 파싱 실패"
  ok "YAML 문법 유효"
else
  echo "  skip: pyyaml 없음 — 문법 검증 생략"
fi

grep -Eq '^[[:space:]]*source:.*scripts/backup_logs\.sh' "$WF" \
  || fail "scp source 에 scripts/backup_logs.sh 가 없음 (EC2 반영이 git pull 에 의존하게 됨)"
ok "scp source 에 backup_logs.sh 포함"

BACKUP_LINE=$(grep -n 'bash scripts/backup_logs\.sh' "$WF" | head -1 | cut -d: -f1)
UP_LINE=$(grep -En 'docker compose .*up -d' "$WF" | head -1 | cut -d: -f1)
[[ -n "$BACKUP_LINE" ]] || fail "ssh 스크립트에 backup_logs.sh 실행이 없음"
[[ -n "$UP_LINE" ]] || fail "ssh 스크립트에 up -d 가 없음"
[[ "$BACKUP_LINE" -lt "$UP_LINE" ]] \
  || fail "백업(${BACKUP_LINE}행)이 up -d(${UP_LINE}행) 뒤에 있음 — 컨테이너 재생성 후 백업은 무의미"
ok "백업 선실행 순서 (backup ${BACKUP_LINE}행 < up -d ${UP_LINE}행)"

sed -n "${BACKUP_LINE}p" "$WF" | grep -q '||' \
  || fail "백업 실행 라인에 || 가드가 없음 — 백업 실패가 배포를 중단시킴"
ok "best-effort 가드 (백업 실패해도 배포 계속)"

echo "PASS: $PASS 검증 통과"
