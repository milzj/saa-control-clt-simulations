#!/usr/bin/env bash
# Run the 'coverage' study for ethanol_fermentation, timed, with stdout+stderr logged.
# Prefers the repo virtualenv (.venv); falls back to the active 'python'.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE="$(basename "$HERE")"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python)"
STAMP="$(date +%Y-%m-%dT%H-%M-%S)"
LOGDIR="$ROOT/logs/$EXAMPLE"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/coverage_$STAMP.log"
echo "[run] $EXAMPLE coverage  ->  $LOG"
cd "$HERE"
# Time the solve and tee stdout+stderr (incl. the 'time' report) into the log.
{ time MPLBACKEND=Agg "$PY" coverage.py "$@" ; } 2>&1 | tee "$LOG"
