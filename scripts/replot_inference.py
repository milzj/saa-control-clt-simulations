"""Re-render the inference figures of saved runs -- no solving.

``inference.py`` saves each algorithm's raw data (per-scenario losses F_i for the
plug-in, the Delta_r statistics for subsampling) to JSON next to the figures.  The
plotters recompute every confidence interval from that raw data, so the figures can
be restyled or re-labelled by replaying the JSON -- the optimizer never runs, and a
whole run re-renders in seconds.

This is model-free (no casadi, no model.py): N, q, r, b, m and the confidence levels
all come from the saved run, so one driver covers every example.

Usage (from the repo root):
    python scripts/replot_inference.py                  # every saved run
    python scripts/replot_inference.py <run_dir> ...    # specific run folders
    python scripts/replot_inference.py --dry-run        # list, write nothing
"""

import argparse
import glob
import os

import ensemblecontrol

from saa_clt.outputs import repo_root

# Every figure is saved in both of these formats (matches the inference drivers).
FORMATS = ("png", "pdf")

PLUGIN_JSON = "plugin.json"
SUB_JSON = "subsampling.json"


def find_run_dirs():
    """Every ``results/<example>/inference/<stamp>/`` folder holding a saved run,
    sorted, oldest stamp first."""
    pattern = os.path.join(repo_root(__file__), "results", "*", "inference", "*")
    return sorted(d for d in glob.glob(pattern)
                  if os.path.isfile(os.path.join(d, PLUGIN_JSON))
                  or os.path.isfile(os.path.join(d, SUB_JSON)))


def _plugin_cis(run):
    """Plug-in CIs recomputed from the raw losses, matching plot_plugin's own
    dispatch on the recorded variance estimator."""
    levels = tuple(run["levels"])
    if run.get("variance") == "out-of-sample":
        return [ensemblecontrol.plugin_oos_ci_from_losses(
            rec["F"], rec["f_opt"], rec["N"], levels) for rec in run["results"]]
    return [ensemblecontrol.plugin_ci_from_losses(rec["F"], rec["f_opt"], levels)
            for rec in run["results"]]


def _subsampling_cis(run):
    levels = tuple(run["levels"])
    return [ensemblecontrol.subsampling_ci_from_deltas(
        rec["deltas"], rec["f_opt"], rec["N"], levels) for rec in run["results"]]


def replot(run_dir, dry_run=False):
    """Re-render whichever algorithms the run folder holds, overwriting in place.

    The folder name IS the run stamp (figures are ``<prefix>_<stamp>_...``), so it is
    passed explicitly -- letting the plotters auto-stamp would scatter a second,
    today-stamped copy of every figure alongside the originals.
    """
    stamp = os.path.basename(os.path.normpath(run_dir))
    plugin_path = os.path.join(run_dir, PLUGIN_JSON)
    sub_path = os.path.join(run_dir, SUB_JSON)
    plugin_run = (ensemblecontrol.load_plugin_run(plugin_path)
                  if os.path.isfile(plugin_path) else None)
    sub_run = (ensemblecontrol.load_subsampling_run(sub_path)
               if os.path.isfile(sub_path) else None)

    # Share the optimal-value y-axis across the two families' _ci{level} plots, but
    # only when both are present -- the same rule the inference drivers apply. The CIs
    # come from the same raw data, so this reproduces the original run's limits.
    value_ylim = None
    if plugin_run is not None and sub_run is not None:
        value_ylim = ensemblecontrol.value_ylim_across(
            _plugin_cis(plugin_run) + _subsampling_cis(sub_run))

    have = [n for n, run in ((PLUGIN_JSON, plugin_run), (SUB_JSON, sub_run))
            if run is not None]
    if dry_run:
        print("[replot] would re-render %s from %s (stamp %s)"
              % (run_dir, " + ".join(have), stamp))
        return []

    written = []
    if plugin_run is not None:
        written += ensemblecontrol.plot_plugin(
            plugin_run, outdir=run_dir, stamp=stamp, value_ylim=value_ylim,
            formats=FORMATS)
    if sub_run is not None:
        written += ensemblecontrol.plot_subsampling(
            sub_run, outdir=run_dir, stamp=stamp, value_ylim=value_ylim,
            formats=FORMATS)
    print("[replot] %s: re-rendered %d figures from %s"
          % (run_dir, len(written), " + ".join(have)))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="*",
                        help="run folders to re-render; default every saved run")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be re-rendered, write nothing")
    args = parser.parse_args()

    run_dirs = args.run_dirs or find_run_dirs()
    if not run_dirs:
        parser.error("no saved inference runs found under results/*/inference/*")
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            parser.error("not a directory: %s" % run_dir)
        replot(run_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
