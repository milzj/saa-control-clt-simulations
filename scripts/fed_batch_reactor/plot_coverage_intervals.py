"""Plot the raw per-replication plug-in CIs saved by the fed-batch coverage study.

Renders the first --n-show of the R intervals in coverage_plugin_intervals.json as one
panel per training size N at a single nominal level, with the reference value
J_hat_ref* across every panel.  Pure post-processing: it reads the saved endpoints and
solves nothing, so it is cheap to re-run with different --level / --n-show.  The figure
is written into the run's own timestamped folder, next to the JSON it came from.

By default it plots the newest coverage run that actually has an intervals file (runs
predating that writer are skipped).

Usage (from the repo root; MPLBACKEND=Agg on headless machines):
    python scripts/fed_batch_reactor/plot_coverage_intervals.py
    ... --level 0.99 --n-show 40
    ... --run-dir results/fed_batch_reactor/coverage/<stamp>
"""

import argparse
import os

from saa_clt.coverage_plotting import (plot_coverage_intervals, plot_interval_grid,
                                       INTERVALS_JSON)
from saa_clt.outputs import latest_run_dir

# Every figure is saved in both of these formats.
FORMATS = ("png", "pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None,
                    help="coverage run folder (or the intervals JSON itself); "
                         "default: the newest run containing %s" % INTERVALS_JSON)
    ap.add_argument("--n-show", type=int, default=100,
                    help="how many of the R replications to plot (default 100, "
                         "clamped to R)")
    ap.add_argument("--level", type=float, default=0.95,
                    help="nominal level to plot; must be one the study ran "
                         "(default 0.95)")
    ap.add_argument("--formats", default=",".join(FORMATS),
                    help="comma-separated figure formats (default %s)"
                         % ",".join(FORMATS))
    ap.add_argument("--no-csv", action="store_true",
                    help="skip the sibling CSV of the plotted intervals")
    ap.add_argument("--per-interval", action="store_true",
                    help="one small panel per replication instead of one panel per N; "
                         "all sizes land in a single figure")
    ap.add_argument("--sizes", default=None,
                    help="comma-separated training sizes for --per-interval "
                         "(default: every size in the run)")
    ap.add_argument("--group-by", choices=("replication", "size"),
                    default="replication",
                    help="--per-interval layout: 'replication' (default) puts every N "
                         "inside each panel; 'size' gives one block of panels per N")
    args = ap.parse_args()

    run = args.run_dir
    if run is None:
        run = latest_run_dir(__file__, "coverage", contains=INTERVALS_JSON)
        if run is None:
            ap.error("no coverage run with %s under results/fed_batch_reactor/"
                     "coverage/ -- run coverage.py first, or pass --run-dir"
                     % INTERVALS_JSON)
    if not os.path.exists(run):
        ap.error("no such run dir or file: %s" % run)

    # Write the figure back into the run's own timestamped folder, whether --run-dir
    # named that folder or the JSON inside it.
    outdir = run if os.path.isdir(run) else os.path.dirname(os.path.abspath(run))

    formats = tuple(f.strip() for f in args.formats.split(","))

    if args.per_interval:
        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
        written = plot_interval_grid(run, sizes=sizes, outdir=outdir,
                                     level=args.level, n_show=args.n_show,
                                     formats=formats, group_by=args.group_by)
    else:
        written = plot_coverage_intervals(
            run, outdir=outdir, level=args.level, n_show=args.n_show,
            formats=formats, save_csv=not args.no_csv)

    for p in written:
        print("[plot] wrote %s" % p, flush=True)


if __name__ == "__main__":
    main()
