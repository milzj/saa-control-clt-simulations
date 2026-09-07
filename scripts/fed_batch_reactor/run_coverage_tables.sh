#!/usr/bin/env bash
# Regenerate both revision coverage tables from stored results; no SAA solves.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"

# Match the revision wrapper's support for a shared checkout environment.
if [ ! -x "$PY" ]; then
  COMMON_GIT="$(git -C "$ROOT" rev-parse --git-common-dir 2>/dev/null || true)"
  if [ -n "$COMMON_GIT" ]; then
    case "$COMMON_GIT" in
      /*) ;;
      *) COMMON_GIT="$ROOT/$COMMON_GIT" ;;
    esac
    COMMON_PY="$(dirname "$COMMON_GIT")/.venv/bin/python"
    if [ -x "$COMMON_PY" ]; then
      PY="$COMMON_PY"
    fi
  fi
fi
if [ ! -x "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  else
    echo "[run] error: no Python interpreter found" >&2
    exit 1
  fi
fi

exec "$PY" "$HERE/coverage_tables.py" "$@"
