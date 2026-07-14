"""Caterpillar plot of the raw per-replication plug-in CIs from a coverage study.

``ensemblecontrol.coverage_study`` reports only aggregate coverage (L/R per N and
level); ``saa_clt.outputs.save_coverage_intervals`` persists the underlying interval
endpoints.  This renders the first ``n_show`` of those R intervals as one panel per
training size N, at a single nominal level, with the reference value J_hat_ref* drawn
across every panel: the intervals visibly contract toward it as N grows (the N^{-1/2}
rate the coverage table only implies), and the ones that miss it are the L/R shortfall
made concrete.

    plot_coverage_intervals
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from ensemblecontrol.inference_plotting import configure_style

from .outputs import load_coverage_intervals
from .postprocess import _savefig_both

__all__ = ["plot_coverage_intervals", "plot_interval_grid"]

INTERVALS_JSON = "coverage_plugin_intervals.json"

# Cover/miss is a two-state encoding, so it must survive colour-blindness and print.
# Okabe-Ito blue/vermillion: worst-case CVD separation dE76 = 91.9 (Machado-2009,
# protanopia), vs 13.3 for a green/red pair; contrast on white 5.19:1 / 3.87:1.  The
# pair is NOT greyscale-separable (dL* = 8.2), so colour is backed by two redundant
# channels -- misses are drawn thicker and take a filled diamond against the covers'
# open circle -- and never carries the distinction alone.
COVER_COLOR = "#0072B2"
MISS_COLOR = "#D55E00"
REF_COLOR = "0.25"


def _level_index(levels, level):
    """Row index of ``level`` in ``levels``; exact match only (no nearest-match, which
    would silently plot a level the caller did not ask for)."""
    for j, lvl in enumerate(levels):
        if abs(lvl - level) < 1e-12:
            return j
    raise ValueError("level %g not in the study's levels %s"
                     % (level, list(levels)))


def _resolve(run_or_path):
    """Accept a run folder or the intervals JSON itself; return (json_path, run_dir)."""
    path = run_or_path
    if os.path.isdir(path):
        path = os.path.join(path, INTERVALS_JSON)
    return path, os.path.dirname(os.path.abspath(path))


def plot_coverage_intervals(run_or_path, outdir=None, prefix="coverage_plugin",
                            stamp=None, level=0.95, n_show=100,
                            formats=("png",), sharey=True, save_csv=True):
    """Plot the first ``n_show`` plug-in CIs per training size N, one panel each.

    ``run_or_path`` is a coverage run folder or its ``coverage_plugin_intervals.json``.
    ``level`` selects one nominal level (must be one the study ran); ``n_show`` is
    clamped to R.  Each interval is drawn as a vertical segment from lo to hi with a
    marker at the centre J_hat_N*, coloured by whether it covers ``f_ref``; panel
    titles carry both the coverage among the shown ``n_show`` and over all R.

    ``sharey`` (default) puts every panel on one y-scale, so the contraction with N is
    visible; without it each panel autoscales and they all look alike.  With
    ``outdir=None`` nothing is written and ``(fig, axes)`` is returned; otherwise the
    figure is saved in each of ``formats`` (plus a sibling CSV of the plotted
    intervals when ``save_csv``) and the list of written paths is returned.
    """
    configure_style()

    path, run_dir = _resolve(run_or_path)
    data = load_coverage_intervals(path)
    if stamp is None:
        stamp = os.path.basename(run_dir)

    levels = data["levels"]
    j = _level_index(levels, level)
    f_ref = data["f_ref"]
    R = data["R"]
    n = min(int(n_show), R)
    blocks = data["results"]

    ncol = 2 if len(blocks) > 1 else 1
    nrow = int(np.ceil(len(blocks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.8 * nrow),
                             sharex=True, sharey=sharey, layout="constrained")
    flat = np.atleast_1d(np.asarray(axes)).ravel()

    x = np.arange(1, n + 1)
    rows = []
    for ax, blk in zip(flat, blocks):
        lo, hi = blk["lo"][j], blk["hi"][j]
        # Coverage over all R first -- the shown subset is a slice of it, so the
        # full-R figure in the title costs nothing extra.
        cov_all = (lo <= f_ref) & (f_ref <= hi)
        lo, hi, cov = lo[:n], hi[:n], cov_all[:n]
        centre = blk["J_hat_N"][:n]

        ax.axhline(f_ref, color=REF_COLOR, ls="--", lw=1.2, zorder=1)
        # Misses last and on top: they are the point of the figure and must never be
        # buried under the covering intervals.
        for mask, color, lw, marker, mfc, ms, z in (
                (cov, COVER_COLOR, 0.9, "o", "none", 2.6, 2),
                (~cov, MISS_COLOR, 1.8, "D", MISS_COLOR, 3.4, 3)):
            if not mask.any():
                continue
            ax.vlines(x[mask], lo[mask], hi[mask], color=color, lw=lw, zorder=z)
            ax.plot(x[mask], centre[mask], ls="none", marker=marker, ms=ms,
                    mfc=mfc, mec=color, mew=0.8, zorder=z + 0.5)

        # Two lines, and R / n_show / the level are stated once in the suptitle rather
        # than per panel: a one-line title carrying all of them overruns the panel and
        # collides with its neighbour.
        ax.set_title("$N = %d$\ncover: %d/%d shown, %.3f of $R$"
                     % (blk["N"], cov.sum(), n, cov_all.mean()), fontsize=10)
        ax.set_xlim(0.5, n + 0.5)

        if save_csv:
            rows.append(np.column_stack([
                np.full(n, blk["N"]), x, centre, blk["se"][:n], lo, hi,
                cov.astype(int)]))

    for ax in flat[len(blocks):]:
        ax.set_visible(False)

    grid = flat.reshape(nrow, ncol)
    for ax in grid[-1, :]:
        ax.set_xlabel("replication")
    for ax in grid[:, 0]:
        ax.set_ylabel(r"$\hat{J}^*_N$")

    handles = [
        mlines.Line2D([], [], color=COVER_COLOR, lw=0.9, marker="o", ms=2.6,
                      mfc="none", mec=COVER_COLOR, label="covers"),
        mlines.Line2D([], [], color=MISS_COLOR, lw=1.8, marker="D", ms=3.4,
                      label="misses"),
        mlines.Line2D([], [], color=REF_COLOR, ls="--", lw=1.2,
                      label=r"$\hat{J}^*_{\mathrm{ref}}$ ($N_{\mathrm{ref}} = %d$)"
                            % data["n_ref"]),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    # Written without a literal "%", which would need escaping under usetex.
    fig.suptitle(r"plug-in CIs at level $1 - \beta = %.2f$: first %d of $R = %d$"
                 " replications" % (level, n, R), fontsize=11)

    if outdir is None:
        return fig, axes

    base = "%s_%s_intervals%d" % (prefix, stamp, int(round(100 * level)))
    written = _savefig_both(fig, os.path.join(outdir, base + ".png"),
                            formats=formats, dpi=130)
    plt.close(fig)

    if save_csv:
        csv_path = os.path.join(outdir, base + ".csv")
        np.savetxt(csv_path, np.vstack(rows),
                   fmt=["%d", "%d", "%.10e", "%.10e", "%.10e", "%.10e", "%d"],
                   delimiter=",", comments="",
                   header="N,replication,J_hat_N,se,lo,hi,covers")
        written.append(csv_path)
    return written


def _block(data, N):
    """The results block for training size ``N``."""
    for blk in data["results"]:
        if blk["N"] == N:
            return blk
    raise ValueError("N = %s not in the study's sample sizes %s"
                     % (N, data["sample_sizes"]))


def _style(covers):
    """(colour, linewidth, marker, marker-facecolor) for a covering / missing CI."""
    if covers:
        return COVER_COLOR, 1.2, "o", "none"
    return MISS_COLOR, 2.0, "D", MISS_COLOR


def _outline_miss(ax):
    """Outline a panel that contains a miss.

    Among hundreds of thumbnails the misses are what the reader scans for, and a
    thin coloured segment alone is too easy to skip.
    """
    for spine in ax.spines.values():
        spine.set_color(MISS_COLOR)
        spine.set_linewidth(1.0)


def _finish_panel(ax, index, f_ref, any_miss):
    ax.axhline(f_ref, color=REF_COLOR, ls="--", lw=0.6, zorder=1)
    ax.set_title(str(index), fontsize=4, pad=1.0)
    ax.tick_params(labelsize=4, length=1.5, pad=0.8)
    if any_miss:
        _outline_miss(ax)


def _grid_shape(n, ncol):
    ncol = int(np.ceil(np.sqrt(n))) if ncol is None else int(ncol)
    return ncol, int(np.ceil(n / ncol))


def _legend_handles(n_ref):
    return [
        mlines.Line2D([], [], color=COVER_COLOR, lw=1.2, marker="o", ms=2.2,
                      mfc="none", mec=COVER_COLOR, label="covers"),
        mlines.Line2D([], [], color=MISS_COLOR, lw=2.0, marker="D", ms=2.2,
                      label="misses (panel outlined)"),
        mlines.Line2D([], [], color=REF_COLOR, ls="--", lw=0.6,
                      label=r"$\hat{J}^*_{\mathrm{ref}}$ ($N_{\mathrm{ref}} = %d$)"
                            % n_ref),
    ]


def plot_interval_grid(run_or_path, sizes=None, outdir=None, prefix="coverage_plugin",
                       stamp=None, level=0.95, n_show=100, formats=("png",),
                       ncol=None, sharey=True, group_by="replication"):
    """One small panel per replication, all training sizes in a single figure.

    The companion to ``plot_coverage_intervals``, which packs all R replications into
    one axes per N.  Here each of the first ``n_show`` replications gets its own panel,
    titled with its replication index, so an individual CI can be read off directly.

    ``group_by`` chooses how the sizes in ``sizes`` (default: every size in the run)
    are arranged:

      * ``"replication"`` (default) -- every N inside each panel, plotted left to right
        along x.  The coverage study draws its scenarios with COMMON RANDOM NUMBERS
        across N (the size-N sample is the first N of one draw), so a panel's intervals
        are nested realisations of the SAME randomness: the contraction with N is
        visible per replication rather than only on average.
      * ``"size"`` -- one block of panels per N, each panel a single interval.  Use it
        to read one N's replications without the neighbouring sizes in the way.

    ``sharey`` (default) puts every panel on one y-scale, so intervals are comparable
    across panels; per-panel autoscaling would stretch each to fill its own axes and
    erase the contraction.  Panels containing a miss are outlined in the miss colour.

    With ``outdir=None`` returns ``(fig, axes)`` where ``axes`` is the flat list of
    drawn panels; otherwise writes the figure in each of ``formats`` and returns the
    written paths.
    """
    configure_style()

    path, run_dir = _resolve(run_or_path)
    data = load_coverage_intervals(path)
    if stamp is None:
        stamp = os.path.basename(run_dir)
    if group_by not in ("replication", "size"):
        raise ValueError("group_by must be 'replication' or 'size', not %r" % group_by)

    j = _level_index(data["levels"], level)
    f_ref = data["f_ref"]
    if sizes is None:
        sizes = list(data["sample_sizes"])
    elif isinstance(sizes, int):
        sizes = [sizes]
    blocks = [_block(data, int(N)) for N in sizes]
    n = min(int(n_show), data["R"])

    if group_by == "replication":
        fig, every = _by_replication(blocks, sizes, j, f_ref, n, ncol, sharey)
    else:
        fig, every = _by_size(blocks, j, f_ref, n, ncol, sharey)

    if sharey and every:
        ylo = min(ax.get_ylim()[0] for ax in every)
        yhi = max(ax.get_ylim()[1] for ax in every)
        for ax in every:
            ax.set_ylim(ylo, yhi)

    fig.legend(handles=_legend_handles(data["n_ref"]), loc="outside lower center",
               ncol=3, frameon=False, fontsize=8)
    tail = (r", $N$ increasing left to right within each panel"
            if group_by == "replication" else "")
    fig.suptitle(r"plug-in CIs at level $1 - \beta = %.2f$: first %d of $R = %d$"
                 " replications, one panel per replication%s"
                 % (level, n, data["R"], tail), fontsize=10)

    if outdir is None:
        return fig, every

    base = "%s_%s_intervals%d_grid%s" % (prefix, stamp, int(round(100 * level)),
                                         "" if group_by == "replication" else "_by_N")
    written = _savefig_both(fig, os.path.join(outdir, base + ".png"),
                            formats=formats, dpi=200)
    plt.close(fig)
    return written


def _by_replication(blocks, sizes, j, f_ref, n, ncol, sharey=True):
    """Panels = replications; every N drawn inside each panel along x."""
    ncol, nrow = _grid_shape(n, ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(0.86 * ncol + 0.6,
                                                  0.86 * nrow + 1.0),
                             sharex=True, sharey=sharey, layout="constrained")
    flat = np.atleast_1d(np.asarray(axes)).ravel()
    xs = np.arange(len(blocks))

    for i, ax in enumerate(flat[:n]):
        any_miss = False
        for k, blk in enumerate(blocks):
            lo, hi = blk["lo"][j][i], blk["hi"][j][i]
            covers = bool(lo <= f_ref <= hi)
            any_miss |= not covers
            color, lw, marker, mfc = _style(covers)
            ax.vlines(k, lo, hi, color=color, lw=lw, zorder=2)
            ax.plot(k, blk["J_hat_N"][i], marker=marker, ms=2.0, mfc=mfc,
                    mec=color, mew=0.6, zorder=3)
        _finish_panel(ax, i + 1, f_ref, any_miss)
        ax.set_xlim(-0.6, len(blocks) - 0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(N) for N in sizes], fontsize=3.5)

    for ax in flat[n:]:
        ax.set_visible(False)
    # No supxlabel: the tick labels already name each N, the suptitle gives the
    # ordering, and it would land on top of the outside legend.
    fig.supylabel(r"$\hat{J}^*_N$", fontsize=9)
    return fig, list(flat[:n])


def _by_size(blocks, j, f_ref, n, ncol, sharey=True):
    """Panels = replications, one block of panels per N; a single interval per panel."""
    ncol, nrow = _grid_shape(n, ncol)
    bcol = 2 if len(blocks) > 1 else 1
    brow = int(np.ceil(len(blocks) / bcol))
    fig = plt.figure(figsize=(0.52 * ncol * bcol + 0.6, 0.56 * nrow * brow + 1.0),
                     layout="constrained")
    subfigs = np.atleast_1d(np.asarray(fig.subfigures(brow, bcol))).ravel()

    every = []
    for sub, blk in zip(subfigs, blocks):
        lo, hi = blk["lo"][j][:n], blk["hi"][j][:n]
        centre = blk["J_hat_N"][:n]
        cov = (lo <= f_ref) & (f_ref <= hi)
        axs = np.atleast_1d(np.asarray(
            sub.subplots(nrow, ncol, sharex=True, sharey=sharey))).ravel()
        for i, ax in enumerate(axs[:n]):
            color, lw, marker, mfc = _style(bool(cov[i]))
            ax.vlines(0, lo[i], hi[i], color=color, lw=lw, zorder=2)
            ax.plot(0, centre[i], marker=marker, ms=2.2, mfc=mfc, mec=color,
                    mew=0.7, zorder=3)
            _finish_panel(ax, i + 1, f_ref, not cov[i])
            ax.set_xlim(-1, 1)
            ax.set_xticks([])               # x carries no information here
        for ax in axs[n:]:
            ax.set_visible(False)
        sub.suptitle(r"$N = %d$   (%d/%d cover)" % (blk["N"], cov.sum(), n),
                     fontsize=9)
        every.extend(axs[:n])

    for sub in subfigs[len(blocks):]:
        sub.set_visible(False)
    fig.supylabel(r"$\hat{J}^*_N$", fontsize=9)
    return fig, every
