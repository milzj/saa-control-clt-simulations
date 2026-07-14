"""CLT histograms with the theoretical N(0, sigma_hat^2) overlay.

``ensemblecontrol.plot_clt`` overlays a normal FITTED to the replicate statistics
themselves (mu = stat.mean(), sd = stat.std()), so that curve tracks the histogram
by construction: it cannot show whether the statistic is actually centered, nor
whether the limit theorem's predicted variance is right.

This local plotter keeps that fitted curve for reference and adds the theory curve

    N(0, sigma_hat_{N_ref}^2),

the limit predicted by

    N^{1/2}(J_hat_N* - J*)  =>  N(0, sigma^2),    sigma^2 = Var F(x*, xi).

sigma_hat_{N_ref} is the Algorithm 1 plug-in estimate taken at the reference solve
(N_ref), whose optimizer x_hat_ref is the best available proxy for x*.  It does not
depend on N, so the SAME curve is drawn on every panel and the histograms should
approach it as N grows.  The drivers capture it during the run and store it in the
saved run's ``meta["sigma_ref"]`` (see ``scripts/<example>/clt.py``); runs saved
before that existed simply omit it and only the fitted curve is drawn.

The statistic is already sqrt(N)-scaled, so the overlay's standard deviation is
sigma_hat itself -- NOT the standard error sigma_hat/sqrt(N) used for CI widths.

Each sample size N gets three figures -- both curves, the fit alone, and the theory
curve alone (see ``VARIANTS``) -- plus a combined all-N overview.
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import norm

from ensemblecontrol import load_clt_run, clt_statistic
# The same publication rcParams ensemblecontrol.plot_clt applies (font size, line
# width, legend), so these histograms match the optimization-bias / monotonicity
# figures rendered beside them in the same run.  Not re-exported by the package's
# __init__, hence the module import; ensemblecontrol is version-pinned, so the path
# is stable.  It sets usetex/serif itself.
from ensemblecontrol.inference_plotting import configure_style

from .postprocess import _savefig_both

__all__ = ["plot_clt_normal"]

_FIT_COLOR = "C0"       # fitted normal (matches ensemblecontrol.plot_clt)
_THEORY_COLOR = "C1"    # N(0, sigma_hat^2) theory curve (orange)

# One figure per variant, per N: both curves overlaid, then each curve alone.
VARIANTS = ("both", "fit", "theory")


def _curve_x(stat, sigma_ref):
    """Plot range covering both the data and the theory curve.

    Taking the union matters: when sigma_hat_ref is much wider (or narrower) than
    the empirical spread -- exactly the mismatch this figure exists to reveal --
    a data-only range would clip the theory curve and hide it.
    """
    lo, hi = float(stat.min()), float(stat.max())
    if sigma_ref:
        lo = min(lo, -3.5 * sigma_ref)
        hi = max(hi, 3.5 * sigma_ref)
    return np.linspace(lo, hi, 400)


def _draw(ax, stat, sigma, show_fit=True, show_theory=True):
    """One panel: density histogram, plus whichever normal curves are requested."""
    stat = np.asarray(stat, dtype=float)
    ax.hist(stat, bins="auto", density=True, color="0.8", edgecolor="0.4")
    xs = _curve_x(stat, sigma)

    sd = float(stat.std())
    if show_fit and sd > 0:
        ax.plot(xs, norm.pdf(xs, float(stat.mean()), sd),
                color=_FIT_COLOR, lw=1.5)
    if show_theory and sigma:
        ax.plot(xs, norm.pdf(xs, 0.0, sigma),
                color=_THEORY_COLOR, lw=1.5, ls="--")

    ax.axvline(0.0, color="k", lw=1.0, ls=":")
    ax.set_xlabel(r"$N^{1/2}(\widehat J_N^* - \widehat{J}_{N_{\mathrm{ref}}}^*)$")


_THEORY_CURVE_LABEL = r"$\mathcal{N}(0,\widehat\sigma_{N_{\mathrm{ref}}}^2)$"
_THEORY_VALUE_LABEL = r"$\widehat\sigma_{N_{\mathrm{ref}}}$"


def _legend_handles(N, N_ref, R, q, r, sigma, show_fit=True, show_theory=True):
    handles = [mpatches.Patch(facecolor="0.8", edgecolor="0.4", label="empirical")]
    if show_fit:
        handles.append(mlines.Line2D([], [], color=_FIT_COLOR, lw=1.5,
                                     label="normal fit"))
    if show_theory and sigma:
        handles.append(mlines.Line2D([], [], color=_THEORY_COLOR, lw=1.5, ls="--",
                                     label=_THEORY_CURVE_LABEL))
    handles.append(mlines.Line2D([], [], color="k", lw=1.0, ls=":", label="0"))
    handles.append(mpatches.Patch(color="none", label=r"$N = %d$" % N))
    handles.append(mpatches.Patch(color="none",
                                  label=r"$N_{\mathrm{ref}} = %d$" % N_ref))
    handles.append(mpatches.Patch(color="none", label=r"$R = %d$" % R))
    if show_theory and sigma:
        handles.append(mpatches.Patch(
            color="none", label=r"%s $= %.4g$" % (_THEORY_VALUE_LABEL, sigma)))
    if q is not None:
        handles.append(mpatches.Patch(color="none", label=r"$q = %d$" % q))
    if r is not None:
        handles.append(mpatches.Patch(color="none", label=r"$r = %g$" % r))
    return handles


def _unify(axes):
    """One common x/y range across panels, so the single N-independent theory
    curve is comparable across N."""
    if len(axes) < 2:
        return
    xlo = min(ax.get_xlim()[0] for ax in axes)
    xhi = max(ax.get_xlim()[1] for ax in axes)
    ylo = min(ax.get_ylim()[0] for ax in axes)
    yhi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)


def plot_clt_normal(run_or_path, outdir, stamp=None, prefix="clt",
                    formats=("png", "pdf"), sigma_ref=None, variants=VARIANTS):
    """Render the CLT histograms with the fitted and/or theory normal curves.

    Drop-in replacement for ``ensemblecontrol.plot_clt`` (same inputs; the
    both-curves figures keep its ``{prefix}_{stamp}_clt_N{N}.*`` /
    ``{prefix}_{stamp}_clt_all.*`` names), adding the N(0, sigma_hat^2) overlay.

    ``run_or_path`` is a saved ``clt.json`` path (or an already-loaded run dict).
    The statistic is recomputed from the stored raw replicate values, so figures
    redraw with no re-solve.

    ``sigma_ref`` is the plug-in sigma_hat at the reference solve -- the CLT's
    asymptotic sigma^2 = Var F(x*, xi) -- and defaults to the run's
    ``meta["sigma_ref"]``, captured by ``scripts/<example>/clt.py``.  It does not
    depend on N, so the same N(0, sigma_hat_{N_ref}^2) is drawn on every panel and
    the histograms should approach it as N grows.  Runs saved before that capture
    existed omit it: only the fitted curve is drawn and a warning is issued.

    ``variants`` selects which per-N figures to emit: ``"both"`` (both curves,
    unsuffixed), ``"fit"`` (``_fit`` suffix) and ``"theory"`` (``_theory`` suffix).
    The combined all-N overview always shows both curves.  Returns written paths.
    """
    configure_style()
    run = load_clt_run(run_or_path) if isinstance(run_or_path, str) else run_or_path
    results = run["results"]
    N_ref, f_ref = run["N_ref"], run["f_ref"]
    q, r = run.get("q"), run.get("r")

    if sigma_ref is None:
        sigma_ref = (run.get("meta") or {}).get("sigma_ref")
    sigma_ref = float(sigma_ref) if sigma_ref else None
    if not sigma_ref:
        warnings.warn(
            "no sigma_ref in meta (and none passed): drawing the fitted normal only, "
            "without the N(0, sigma_hat^2) theory curve. Re-run clt.py to capture it.",
            stacklevel=2)

    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError("unknown variant(s) %s; expected a subset of %s"
                         % (sorted(unknown), list(VARIANTS)))
    if not sigma_ref:
        # A theory-only panel with no sigma would be a bare histogram; drop it.
        variants = tuple(v for v in variants if v != "theory")

    stats = [clt_statistic(rec["values"], rec["N"], f_ref) for rec in results]
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, prefix if stamp is None
                        else "{}_{}".format(prefix, stamp))

    # Every figure is built and kept open before saving, so the whole set -- each
    # variant's per-N panels plus the combined overview -- shares one x/y range.
    show = {"both": (True, True), "fit": (True, False), "theory": (False, True)}
    suffix = {"both": "", "fit": "_fit", "theory": "_theory"}

    pending = []    # (fig, ax, savepath)
    for variant in variants:
        show_fit, show_theory = show[variant]
        for rec, stat in zip(results, stats):
            fig, ax = plt.subplots()
            _draw(ax, stat, sigma_ref, show_fit=show_fit, show_theory=show_theory)
            ax.set_ylabel("density")
            ax.legend(handles=_legend_handles(
                rec["N"], N_ref, stat.size, q, r, sigma_ref, show_fit=show_fit,
                show_theory=show_theory))
            pending.append((fig, ax, "{}_clt_N{}{}.png".format(
                base, rec["N"], suffix[variant])))

    fig_all, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.4))
    axes = np.atleast_1d(axes)
    for ax, rec, stat in zip(axes, results, stats):
        _draw(ax, stat, sigma_ref)
        ax.legend(handles=_legend_handles(rec["N"], N_ref, stat.size, q, r,
                                          sigma_ref), fontsize="small")
    axes[0].set_ylabel("density")
    fig_all.suptitle(r"$N^{1/2}(\widehat J_N^* - \widehat{J}_{N_{\mathrm{ref}}}^*)$")
    pending.append((fig_all, None, "{}_clt_all.png".format(base)))

    if len(results) >= 2:
        _unify([ax for _fig, ax, _p in pending if ax is not None] + list(axes))

    written = []
    for fig, _ax, savepath in pending:
        fig.tight_layout()
        written += _savefig_both(fig, savepath, formats=formats, dpi=130)
        plt.close(fig)
    return written
