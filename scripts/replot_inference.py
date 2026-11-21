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
    python scripts/replot_inference.py --subsampling-only <run_dir> ...
    python scripts/replot_inference.py --reference <clt_reference.json> <run_dir>
    python scripts/replot_inference.py --dry-run        # list, write nothing
"""

import argparse
from contextlib import contextmanager
import glob
import importlib
import json
import os

import matplotlib

matplotlib.use("Agg")

import ensemblecontrol
import matplotlib.pyplot as plt
from ensemblecontrol.inference_plotting import configure_style
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from saa_clt.outputs import repo_root
from saa_clt.subsampling_width_mc import subsampling_cis_from_deltas

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
    return [subsampling_cis_from_deltas(
        rec["deltas"], rec["f_opt"], rec["N"], levels) for rec in run["results"]]


@contextmanager
def _exact_subsampling_plotter():
    """Temporarily give EnsembleControl's plotter the exact-rank CI helper."""
    plotting = importlib.import_module("ensemblecontrol.inference_plotting")
    original = plotting.subsampling_ci_from_deltas
    plotting.subsampling_ci_from_deltas = subsampling_cis_from_deltas
    try:
        yield
    finally:
        plotting.subsampling_ci_from_deltas = original


def _plot_subsampling(*args, **kwargs):
    with _exact_subsampling_plotter():
        return ensemblecontrol.plot_subsampling(*args, **kwargs)


def _plot_reference_sweeps(runs, run_dir, stamp, value_ylim, reference_value):
    sweeps = []
    legend_top = 0.0
    reference_label = r"$\widehat J_{N_{\mathrm{ref}}}^*$"
    for prefix, run in runs:
        levels = run["levels"]
        if prefix == "subsampling":
            panels = _plot_subsampling(run, band_level=levels[0], value_ylim=value_ylim)
        else:
            panels = ensemblecontrol.plot_plugin(run, value_ylim=value_ylim)

        try:
            # The figure-returning API also returns histograms and log-log width plots.
            interval_panels = ((fig, ax) for fig, ax in panels if isinstance(ax, Axes)
                               and ax.get_xscale() == "log" and ax.get_yscale() == "linear")
            for level, (fig, ax) in zip(levels, interval_panels, strict=True):
                original_legend = ax.get_legend()
                original_labels = [text.get_text() for text in original_legend.get_texts()]
                data_handles, _ = ax.get_legend_handles_labels()
                estimate = data_handles[0]
                if isinstance(estimate, ErrorbarContainer):
                    estimate = estimate.lines[0]
                estimate_dot = Line2D(
                    [], [], marker="o", linestyle="none", color=estimate.get_color(),
                    markersize=estimate.get_markersize(),
                )
                reference_line = ax.axhline(reference_value, color="tab:brown",
                                            ls="--", lw=1.6, zorder=1)
                legend = ax.legend(
                    [estimate_dot, reference_line] + original_legend.legend_handles[1:],
                    [original_labels[0], reference_label] + original_labels[1:],
                    loc="lower right",
                )
                bounds = legend.get_window_extent(fig.canvas.get_renderer())
                legend_top = max(legend_top, ax.transAxes.inverted().transform_bbox(bounds).ymax)
                sweeps.append((prefix, level, fig, ax))
        finally:
            for fig, _ in panels:
                plt.close(fig)

    # Keep the original data range above the tallest legend in every panel.
    lo, hi = value_ylim
    reserve = legend_top + 0.025
    if reserve >= 1:
        raise ValueError("The reference legend is too tall to fit inside the axes")
    lower_limit = lo - (hi - lo) * reserve / (1 - reserve)
    written = []
    for prefix, level, fig, ax in sweeps:
        ax.set_ylim(lower_limit, hi)
        for fmt in FORMATS:
            filename = f"{prefix}_{stamp}_ci{100 * level:.0f}_reference.{fmt}"
            path = os.path.join(run_dir, filename)
            fig.savefig(path, dpi=130)
            written.append(path)
    return written


def replot(run_dir, dry_run=False, subsampling_only=False, reference_path=None):
    """Re-render whichever algorithms the run folder holds, overwriting in place.

    The folder name IS the run stamp (figures are ``<prefix>_<stamp>_...``), so it is
    passed explicitly -- letting the plotters auto-stamp would scatter a second,
    today-stamped copy of every figure alongside the originals.
    With ``reference_path``, only interval sweeps are saved, with a separate
    ``_reference`` suffix.
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
        suffix = " (subsampling figures only)" if subsampling_only else ""
        if reference_path is not None:
            suffix += f" (interval sweeps with reference from {reference_path})"
        print("[replot] would re-render %s from %s (stamp %s)%s"
              % (run_dir, " + ".join(have), stamp, suffix))
        return []

    if subsampling_only and sub_run is None:
        raise ValueError("--subsampling-only requires a subsampling.json file")

    if reference_path is not None:
        if not have:
            raise ValueError("--reference requires a plugin.json or subsampling.json file")
        with open(reference_path) as fh:
            reference = json.load(fh)
        if value_ylim is None:
            cis = _plugin_cis(plugin_run) if plugin_run is not None else _subsampling_cis(sub_run)
            value_ylim = ensemblecontrol.value_ylim_across(cis)
        lo, hi = value_ylim
        pad = 0.05 * (hi - lo)
        reference_value = reference["f_ref"]
        value_ylim = (min(lo, reference_value - pad), max(hi, reference_value + pad))

        runs = [] if subsampling_only else [("plugin", plugin_run)]
        runs.append(("subsampling", sub_run))
        configure_style()
        with plt.rc_context({"text.usetex": False, "font.serif": ["DejaVu Serif"]}):
            written = _plot_reference_sweeps(
                [(prefix, run) for prefix, run in runs if run is not None],
                run_dir, stamp, value_ylim, reference_value)
        print(f"[replot] {run_dir}: saved {len(written)} figures with "
              f"the {reference['n_ref']}-scenario reference from {reference_path}")
        return written

    if subsampling_only:
        written = _plot_subsampling(
            sub_run,
            outdir=run_dir,
            stamp=stamp,
            value_ylim=value_ylim,
            formats=FORMATS,
        )
        print("[replot] %s: re-rendered %d subsampling figures from %s"
              % (run_dir, len(written), SUB_JSON))
        return written

    written = []
    if plugin_run is not None:
        written += ensemblecontrol.plot_plugin(
            plugin_run, outdir=run_dir, stamp=stamp, value_ylim=value_ylim,
            formats=FORMATS)
    if sub_run is not None:
        written += _plot_subsampling(
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
    parser.add_argument(
        "--subsampling-only",
        action="store_true",
        help="rewrite only the PDF/PNG subsampling figures",
    )
    parser.add_argument("--reference", metavar="PATH",
                        help="include the stored CLT reference in CI sweeps for one run folder")
    args = parser.parse_args()

    if args.reference is not None and len(args.run_dirs) != 1:
        parser.error("--reference requires exactly one run folder")
    run_dirs = args.run_dirs or find_run_dirs()
    if not run_dirs:
        parser.error("no saved inference runs found under results/*/inference/*")
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            parser.error("not a directory: %s" % run_dir)
        replot(
            run_dir,
            dry_run=args.dry_run,
            subsampling_only=args.subsampling_only,
            reference_path=args.reference,
        )


if __name__ == "__main__":
    main()
