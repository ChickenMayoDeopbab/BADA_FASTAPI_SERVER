#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMPOSE_DIR=${COMPOSE_DIR:-$(dirname "$SCRIPT_DIR")}
SERVICE=${SERVICE:-app}
S3_PREFIX=${S3_PREFIX:-logs/${SERVICE}}
STATE_FILE=${STATE_FILE:-$HOME/.bada/last_log_backup_${SERVICE}}
LOCK_DIR=${STATE_FILE}.lock

log() { echo "[backup_logs] $*"; }

ENV_FILE="$COMPOSE_DIR/.env"
env_val() { sed -n "s/^[[:space:]]*$1=//p" "$ENV_FILE" | tail -1 | tr -d '\r' | tr -d '"' | tr -d "'"; }

if [[ -f "$ENV_FILE" ]]; then
  if [[ -z "${S3_BUCKET:-}" ]]; then
    S3_BUCKET=$(env_val S3_BUCKET)
  fi
  if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    _key=$(env_val AWS_ACCESS_KEY)
    _secret=$(env_val AWS_SECRET_KEY)
    if [[ -n "$_key" && -n "$_secret" ]]; then
      export AWS_ACCESS_KEY_ID="$_key"
      export AWS_SECRET_ACCESS_KEY="$_secret"
    fi
  fi
  if [[ -z "${AWS_DEFAULT_REGION:-}" ]]; then
    _region=$(env_val AWS_REGION)
    if [[ -n "$_region" ]]; then
      export AWS_DEFAULT_REGION="$_region"
    fi
  fi
fi
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-ap-northeast-2}

if [[ -z "${S3_BUCKET:-}" ]]; then
  log "S3_BUCKET 이 없다 (환경변수 또는 $ENV_FILE)" >&2
  exit 1
fi

mkdir -p "$(dirname "$STATE_FILE")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log "이미 실행 중 ($LOCK_DIR) — 종료"
  exit 0
fi
TMP=""
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if [[ -n "$TMP" ]]; then rm -f "$TMP" "$TMP.gz" 2>/dev/null || true; fi
}
trap cleanup EXIT

if [[ -z "${CONTAINER:-}" ]]; then
  CONTAINER=$(cd "$COMPOSE_DIR" && docker compose ps -q "$SERVICE" | head -1)
fi
if [[ -z "$CONTAINER" ]]; then
  log "실행 중인 '$SERVICE' 컨테이너가 없다 — 백업 생략"
  exit 0
fi

NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SINCE=""
if [[ -f "$STATE_FILE" ]]; then
  SINCE=$(cat "$STATE_FILE")
fi

TMP=$(mktemp -t bada-logs.XXXXXX)
if [[ -n "$SINCE" ]]; then
  docker logs --since "$SINCE" --until "$NOW" "$CONTAINER" >"$TMP" 2>&1
else
  docker logs --until "$NOW" "$CONTAINER" >"$TMP" 2>&1
fi

if [[ ! -s "$TMP" ]]; then
  echo "$NOW" >"$STATE_FILE"
  log "새 로그 없음 (${SINCE:-처음} ~ $NOW)"
  exit 0
fi

gzip -f "$TMP"
DATE_PATH="${NOW:0:4}/${NOW:5:2}/${NOW:8:2}"
FROM_LABEL=${SINCE:-container-start}
KEY="$S3_PREFIX/$DATE_PATH/${SERVICE}_${FROM_LABEL//:/-}_${NOW//:/-}.log.gz"

aws s3 cp "$TMP.gz" "s3://$S3_BUCKET/$KEY" --only-show-errors

echo "$NOW" >"$STATE_FILE"
log "업로드 완료: s3://$S3_BUCKET/$KEY"
