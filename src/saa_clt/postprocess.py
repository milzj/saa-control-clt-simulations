"""Control post-processing plots shared by both examples.

Plots the direct (solved) control against the singular control reconstructed by
the model-agnostic engine in ``ensemblecontrol.singular_control``
(``detect_arcs``, ``build_switching_machinery``, ``ensemble_phi_and_control``),
plus nominal-vs-SAA overlays.  The two examples share this one copy; the
fed-batch example simply never passes ``case``.  Every figure is written to the
explicit ``outdir``/``savepath`` the caller passes (a
``results/<example>/...`` run folder).

    plot_phi_constructed_control, plot_postprocessed_controls, plot_direct_controls
"""

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ensemblecontrol import (detect_arcs, build_switching_machinery,
                             ensemble_phi_and_control)

_FONTS_CONFIGURED = False


def _configure_fonts():
    # Use LaTeX (serif Computer Modern) when a latex binary is on PATH; otherwise
    # keep matplotlib's default mathtext so plots still render without TeX. Runs
    # once per process.
    global _FONTS_CONFIGURED
    if _FONTS_CONFIGURED:
        return
    _FONTS_CONFIGURED = True
    if shutil.which("latex") is not None:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsfonts}",
            "font.family": "serif",
        })


# ========================================================================== #
# Post-processing plotters (ethanol-fermentation demo; engine in ensemblecontrol)
# ========================================================================== #
def _nqr_label(N=None, q=None, r=None):
    """Legend annotation ``($N=..., q=..., r=...$)`` from whichever of the sample
    size N, control-mesh size q, and scenario radius r are provided."""
    parts = []
    if N is not None:
        parts.append("N={}".format(N))
    if q is not None:
        parts.append("q={}".format(q))
    if r is not None:
        parts.append("r={}".format(r))
    return (r"($%s$)" % ", ".join(parts)) if parts else None


def _savefig_both(fig, savepath, formats=("png", "pdf"), **savefig_kw):
    """Save ``fig`` once per format (default PNG + PDF), swapping the extension of
    ``savepath``.  Returns the list of written paths."""
    root = os.path.splitext(savepath)[0]
    written = []
    for fmt in formats:
        p = "{}.{}".format(root, fmt)
        fig.savefig(p, bbox_inches="tight", **savefig_kw)
        written.append(p)
    return written


def plot_phi_constructed_control(model, samples, w_opt, prefix, outdir="output",
                                 max_samples=None, S=1, stamp=None, case=None):
    """From a solved single-shooting control w_opt and the parameter `samples`
    (nominal = one vector, risk-neutral = an ensemble), reconstruct the
    (ensemble) singular control via phi_j = -sum_s A_s / sum_s B_s using per-
    sample backward adjoints, and plot it against the solved control together
    with the ensemble switching function.  Writes <outdir>/<stem>_phi.png, where
    <stem> = <prefix> (or <prefix>_<stamp> when a timestamp is given).

    `case` labels the standalone postprocessed-control plot as
    "postprocessed <case> control" (e.g. "postprocessed SAA control"); when None
    the legend falls back to the generic "postprocessed control".
    """
    _configure_fonts()
    stem = prefix if stamp is None else "%s_%s" % (prefix, stamp)
    N = model.nintervals
    lb = model.control_bounds[0][0]; ub = model.control_bounds[1][0]
    u_ws = np.asarray(w_opt, float).ravel()[:N]
    smp = np.atleast_2d(np.asarray(samples, float))
    if max_samples is not None and smp.shape[0] > max_samples:
        idx = np.linspace(0, smp.shape[0] - 1, max_samples).round().astype(int)
        print("[phi] %s: subsampling ensemble %d -> %d samples for the diagnostic"
              % (prefix, smp.shape[0], max_samples))
        smp = smp[idx]
    k_samples = [smp[i] for i in range(smp.shape[0])]
    mach = build_switching_machinery(model)
    arcs = detect_arcs(u_ws, lb, ub, model.mesh_width)
    t, phi, ucon, sig, C_direct, C_post = ensemble_phi_and_control(
        model, mach, k_samples, arcs, lb, ub, u_ws, S=S)
    t_ws = np.linspace(0.0, model.final_time, N + 1)
    ens = "ensemble" if smp.shape[0] > 1 else "nominal"

    fig, (axc, axs) = plt.subplots(2, 1, figsize=(9, 7.0), sharex=True)
    axc.step(t_ws, np.append(u_ws, u_ws[-1]), where="post", color="0.6", lw=1.1,
             label="solved control (%d samples)" % smp.shape[0])
    axc.plot(t, ucon, color="tab:blue", lw=1.6,
             label=r"constructed control via %s $\phi_j$ (backward adjoint)" % ens)
    axc.set_ylim(lb - 0.3, ub + 0.3)
    axc.set_ylabel("control u")
    axc.set_title(r"%s: constructed control via %s $\phi_j$" % (prefix, ens))
    axc.legend(loc="best", fontsize=8)
    slim = 1.2 * float(np.nanmax(np.abs(sig))) if np.any(np.isfinite(sig)) else 1.0
    axs.plot(t, sig, color="tab:red", lw=1.2,
             label=r"$\sigma=\sum_s p^{(s)T}b(x^{(s)})$")
    axs.axhline(0.0, color="0.5", lw=0.8)
    axs.set_ylim(-slim, slim)
    axs.set_xlabel("time")
    axs.set_ylabel(r"switching function $\sigma$")
    axs.legend(loc="best", fontsize=8)
    for ax in (axc, axs):
        for a in arcs:
            ax.axvline(a["t0"], color="0.85", lw=0.6, zorder=0)
            if a["type"] == "SING":
                ax.axvspan(a["t0"], a["t1"], color="tab:blue", alpha=0.08, zorder=0)
    fig.tight_layout()
    pth = _savefig_both(fig, os.path.join(outdir, stem + "_phi.png"),
                        dpi=130)[0]
    plt.close("all")

    # standalone postprocessed control (same layout as the *_control.png plots).
    radius = getattr(model, "perturbation_radius", None)
    fig2, ax2 = plt.subplots()
    y = np.concatenate([[np.nan], ucon[1:]])        # piecewise-constant, where='pre'
    ctrl_label = ("postprocessed %s control" % case if case is not None
                  else "postprocessed control")
    ax2.step(t, y, where="pre", label=ctrl_label)
    ax2.set_xlabel(r"$t$"); ax2.grid()
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(mpatches.Patch(color="none"))
    labels.append(_nqr_label(N=smp.shape[0], q=N, r=radius))
    ax2.legend(handles, labels)
    pth2 = _savefig_both(fig2, os.path.join(outdir,
                         stem + "_postprocessed_control.png"))[0]
    plt.close(fig2)

    # save the postprocessed control samples (t, u) alongside the plot
    csv = os.path.join(outdir, stem + "_postprocessed_control.csv")
    np.savetxt(csv, np.column_stack([t, ucon]), delimiter=",",
               header="t,u", comments="")

    # objective values: direct vs postprocessed control (SAA mean C)
    diff = C_post - C_direct
    rel = diff / abs(C_direct) if C_direct != 0 else float("nan")
    lines = [
        "%s objective  (SAA mean C = mean_s [F(x^(s)(T))] = mean of the model "
        "final cost,  N=%d samples)" % (prefix, smp.shape[0]),
        "  direct control        C = %+.10f   (reward %.10f)" % (C_direct, -C_direct),
        "  postprocessed (phi)   C = %+.10f   (reward %.10f)" % (C_post, -C_post),
        "  difference (post - direct) = %+.6e" % diff,
        "  relative difference        = %+.6e   (%+.4f %%)" % (rel, 100.0 * rel),
    ]
    txt = os.path.join(outdir, stem + "_objective.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    for ln in lines:
        print("[phi] " + ln)
    print("[phi] wrote %s , %s , %s , %s   (%d arcs: %s)"
          % (pth, pth2, csv, txt, len(arcs), [a["type"] for a in arcs]))

    # return the constructed control on its grid so callers can overlay several
    # postprocessed controls (e.g. nominal vs risk-neutral) in one figure
    return t, ucon


def plot_postprocessed_controls(cases, savepath, N=None, q=None, r=None,
                                alpha=None, save_csv=True):
    """Overlay several postprocessed (constructed) controls in one figure.

    cases: list of (t, ucon, label) as returned by plot_phi_constructed_control.
    The legend annotates the sample size N, control-mesh size q, and scenario
    radius r; ``alpha`` is a legacy fallback (older demos annotate the
    running-cost weight instead).  If save_csv, also writes the controls next to
    the plot as <savepath without ext>.csv.
    """
    _configure_fonts()
    fig, ax = plt.subplots()
    for t, ucon, label in cases:
        y = np.concatenate([[np.nan], np.asarray(ucon)[1:]])  # piecewise-constant
        ax.step(t, y, where="pre", label=label)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"postprocessed control $u$")
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    note = _nqr_label(N=N, q=q, r=r)
    if note is None and alpha is not None:
        note = r"($\alpha={}$)".format(alpha)
    if note is not None:
        handles.append(mpatches.Patch(color="none"))
        labels.append(note)
    ax.legend(handles, labels)
    _savefig_both(fig, savepath)
    plt.close(fig)
    print("[phi] wrote %s (png, pdf)" % savepath)

    if save_csv:
        # share the time grid if the cases agree; else save each case's own t
        cols, header = [], []
        t0 = np.asarray(cases[0][0])
        same_grid = all(np.asarray(t).shape == t0.shape and np.allclose(t, t0)
                        for t, _, _ in cases)
        if same_grid:
            cols.append(t0); header.append("t")
            for _, ucon, label in cases:
                cols.append(np.asarray(ucon)); header.append("u_" + str(label))
        else:
            for t, ucon, label in cases:
                cols += [np.asarray(t), np.asarray(ucon)]
                header += ["t_" + str(label), "u_" + str(label)]
        csv = os.path.splitext(savepath)[0] + ".csv"
        np.savetxt(csv, np.column_stack(cols), delimiter=",",
                   header=",".join(header), comments="")
        print("[phi] wrote %s" % csv)


def plot_direct_controls(cases, savepath, N=None, q=None, r=None, alpha=None,
                         save_csv=True):
    """Overlay several direct (solved) piecewise-constant controls in one figure.

    cases: list of (tgrid, u, label), where tgrid has N+1 node times and u has N
    interval values (u_k held over (t_k, t_{k+1}], rendered where='pre').  The
    legend annotates the sample size N, control-mesh size q, and scenario radius
    r; ``alpha`` is a legacy fallback (older demos annotate the running-cost
    weight instead).  If save_csv, also writes the controls next to the plot as
    <savepath without ext>.csv (interval left-edge time + u).
    """
    _configure_fonts()
    fig, ax = plt.subplots()
    for tgrid, u, label in cases:
        y = np.concatenate([[np.nan], np.asarray(u)])  # hold u_k over (t_k, t_{k+1}]
        ax.step(tgrid, y, where="pre", label=label)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"control $u$")
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    note = _nqr_label(N=N, q=q, r=r)
    if note is None and alpha is not None:
        note = r"($\alpha={}$)".format(alpha)
    if note is not None:
        handles.append(mpatches.Patch(color="none"))
        labels.append(note)
    ax.legend(handles, labels)
    _savefig_both(fig, savepath)
    plt.close(fig)
    print("[phi] wrote %s (png, pdf)" % savepath)

    if save_csv:
        # interval left-edge times; share the grid if the cases agree
        cols, header = [], []
        t0 = np.asarray(cases[0][0])[:-1]
        same_grid = all(np.asarray(tg)[:-1].shape == t0.shape
                        and np.allclose(np.asarray(tg)[:-1], t0)
                        for tg, _, _ in cases)
        if same_grid:
            cols.append(t0); header.append("t")
            for _, u, label in cases:
                cols.append(np.asarray(u)); header.append("u_" + str(label))
        else:
            for tg, u, label in cases:
                cols += [np.asarray(tg)[:-1], np.asarray(u)]
                header += ["t_" + str(label), "u_" + str(label)]
        csv = os.path.splitext(savepath)[0] + ".csv"
        np.savetxt(csv, np.column_stack(cols), delimiter=",",
                   header=",".join(header), comments="")
        print("[phi] wrote %s" % csv)
