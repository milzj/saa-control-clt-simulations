#!/usr/bin/env bash
# Run all four studies for fed_batch_reactor in order (each timed + logged separately).
# NOTE: the coverage study can take a long time (hundreds of SAA solves).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/run_nominal_saa.sh"
"$HERE/run_clt.sh"
"$HERE/run_inference.sh"
# Coverage not included in paper
# "$HERE/run_coverage.sh"
