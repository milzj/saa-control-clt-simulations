#!/usr/bin/env bash
# Run the conditional repeated-subsampling b_N width diagnostic, timed and logged.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE="$(basename "$HERE")"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"

# Codex worktrees intentionally omit the main checkout's virtual environment.
# The common Git directory identifies that checkout without hard-coding it.
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

STAMP="$(date +%Y-%m-%dT%H-%M-%S)"
LOGDIR="$ROOT/logs/$EXAMPLE"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/subsampling_width_monte_carlo_$STAMP.log"
CACHE_DIR="${TMPDIR:-/tmp}/saa-control-clt-cache"
mkdir -p "$CACHE_DIR/matplotlib" "$CACHE_DIR/xdg"

# Outer processes supply the parallelism.  Keep every numerical kernel in each
# worker single-threaded to avoid oversubscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[run] $EXAMPLE repeated-subsampling width -> $LOG"
cd "$ROOT"
{ time "$PY" "$HERE/subsampling_width_monte_carlo.py" "$@"; } 2>&1 | tee "$LOG"
