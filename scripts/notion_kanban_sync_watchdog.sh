#!/usr/bin/env bash
set -euo pipefail
PY=${HERMES_NOTION_SYNC_PYTHON:-/home/solo/.hermes/hermes-agent/venv/bin/python}
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
SCRIPT=${HERMES_NOTION_SYNC_SCRIPT:-$HOME/.hermes/profiles/dev/scripts/notion_kanban_sync.py}
REPORT_DIR=${HERMES_NOTION_SYNC_REPORT_DIR:-$HOME/.hermes/reports/hermes-notion-sync}
MAX_CREATES=${HERMES_NOTION_SYNC_MAX_CREATES:-25}
exec "$PY" "$SCRIPT" --apply --quiet --max-creates "$MAX_CREATES" --report-dir "$REPORT_DIR"
