"""Nominal and risk-neutral (SAA) solutions for the fed-batch reactor (Ipopt).

Solves the nominal problem and the risk-neutral SAA problem on a fine 2000-interval
mesh, plots the state and control trajectories, and reconstructs the singular
control by phi-postprocessing (see saa_clt.postprocess).

Outputs go to results/fed_batch_reactor/controls-states/<stamp>/{nominal,saa}/
(per-solution control and state plots) with the nominal-vs-SAA overlays at the
run-folder top level; every figure is saved as both PNG and PDF.

Run from the repo root (or via scripts/fed_batch_reactor/run_nominal_saa.sh):
    MPLBACKEND=Agg python scripts/fed_batch_reactor/nominal_saa.py
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import ensemblecontrol
from saa_clt.outputs import study_dir
from saa_clt.postprocess import (plot_phi_constructed_control,
                                 plot_postprocessed_controls,
                                 plot_direct_controls)
from model import FedBatchReactor
from config import (SOLUTION_NINTERVALS, SAA_SAMPLES, SAA_SAMPLER_SEED,
                    ipopt_options, TOL_SOLUTION)

# Fine mesh for the nominal / risk-neutral solutions (config).
NINTERVALS = SOLUTION_NINTERVALS

# phi postprocessing grid: number of sub-intervals per direct-solve interval.
# S=1 reconstructs on the direct-solve grid; increase (e.g. 4, 16) for a finer grid.
POSTPROC_S = 1

# Every figure is saved in both of these formats.
FORMATS = ("png", "pdf")


def solve(model, samples):
    # Ipopt single-shooting SAA solve at the nominal/SAA tolerance (1e-8).
    saa_problem = ensemblecontrol.SAAProblem(
        model, samples, MultipleShooting=False,
        ipopt_options=ipopt_options(TOL_SOLUTION))
    w_opt, f_opt = saa_problem.solve()
    return saa_problem, w_opt


def _save_both(fig, base_no_ext):
    for fmt in FORMATS:
        fig.savefig("%s.%s" % (base_no_ext, fmt), bbox_inches="tight")


def _plot_states(plotter, case, base, nsigma):
    # per-state trajectories with a +/- nsigma ensemble band; clean "<case> x{j}"
    # legends and "<base>_x{j}.{png,pdf}" filenames (decoupled so the label can
    # carry a space while the filename does not).
    radius = plotter.saa_problem.control_problem.perturbation_radius
    figs, _ = plotter.plot_states(
        per_state=True, radius=radius, nsigma=nsigma,
        state_labels=["%s x%d" % (case, j) for j in range(plotter.nstates)])
    for j, fig in enumerate(figs):
        _save_both(fig, "%s_x%d" % (base, j))
        plt.close(fig)


def plot(saa_problem, w_opt, file_prefix, case, case_dir, stamp, nsigma=1):
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    radius = saa_problem.control_problem.perturbation_radius
    base = os.path.join(case_dir, "%s_%s" % (file_prefix, stamp))

    plotter.plot_controls(control_labels=["%s control" % case], step=True,
                          radius=radius, savepath=base + "_control.png",
                          formats=FORMATS)
    plt.close("all")
    _plot_states(plotter, case, base, nsigma)

    # save the direct (solved) control(s): interval left-edge time + value/control
    t_left = plotter.tgrid[:-1]
    cols = [t_left] + [plotter.controls[c] for c in range(plotter.ncontrols)]
    labels = (["u"] if plotter.ncontrols == 1
              else ["u%d" % c for c in range(plotter.ncontrols)])
    np.savetxt(base + "_control.csv", np.column_stack(cols),
               delimiter=",", header=",".join(["t"] + labels), comments="")

    # return the direct (solved) control for overlaying nominal vs SAA
    return plotter.tgrid, plotter.controls[0]


def plot_postprocessed_states(saa_problem, ucon, case, case_dir, stamp, nsigma):
    """Plot the ensemble states obtained by rolling the postprocessed (phi-
    reconstructed) control through the dynamics -- reconstructed with the SAME
    integrator as the direct states, so the two are directly comparable."""
    u_post = np.asarray(ucon, float).ravel()[1:]      # per-interval control (where='pre')
    if u_post.size != saa_problem.control_problem.nintervals:
        return   # only when the postprocessed grid matches the solve grid (S=1)
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, u_post)
    base = os.path.join(case_dir, "%s_%s" % ("saa_postprocessed", stamp))
    _plot_states(plotter, case, base, nsigma)


if __name__ == "__main__":

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # One timestamped run folder per solve:
    # results/fed_batch_reactor/controls-states/<stamp>/{nominal,saa}/
    run_dir = study_dir(__file__, "controls-states", stamp)
    NOMINAL_DIR = os.path.join(run_dir, "nominal")
    SAA_DIR = os.path.join(run_dir, "saa")
    for _d in (NOMINAL_DIR, SAA_DIR):
        os.makedirs(_d, exist_ok=True)

    # -- Nominal problem -----------------------------------------------------
    nominal_model = FedBatchReactor()
    nominal_model.nintervals = NINTERVALS
    nominal_param = nominal_model.nominal_param
    saa_nom, w_nom = solve(nominal_model, nominal_param)
    td_nom, ud_nom = plot(saa_nom, w_nom, "nominal", "nominal", NOMINAL_DIR, stamp)
    # constructed control via nominal phi_j (switching structure from this run)
    t_nom, u_nom = plot_phi_constructed_control(
        nominal_model, nominal_param, w_nom, "nominal", outdir=NOMINAL_DIR,
        S=POSTPROC_S, stamp=stamp)

    # -- Risk-neutral (SAA) problem ------------------------------------------
    saa_model = FedBatchReactor()
    saa_model.nintervals = NINTERVALS

    # sampler: relative multiplicative perturbation of the nominal parameters,
    # xi_j = (1 + r * U[-1, 1]) * nominal_j, with radius r from the model
    # (FedBatchReactor.scenario_sampler) so it matches every other driver.
    N = SAA_SAMPLES
    samples = saa_model.scenario_sampler(seed=SAA_SAMPLER_SEED).sample(N)

    saa_rn, w_rn = solve(saa_model, samples)
    td_rn, ud_rn = plot(saa_rn, w_rn, "saa", "SAA", SAA_DIR, stamp, nsigma=3)
    # constructed control via ensemble phi_j (switching structure from this run)
    t_rn, u_rn = plot_phi_constructed_control(
        saa_model, samples, w_rn, "saa", outdir=SAA_DIR, S=POSTPROC_S,
        stamp=stamp)
    # states obtained by rolling the postprocessed (phi) control through the dynamics
    plot_postprocessed_states(saa_rn, u_rn, "SAA postprocessed", SAA_DIR, stamp,
                              nsigma=3)

    # -- Nominal vs SAA overlays (at the run-folder top level) ----------------
    q = saa_model.nintervals
    r = saa_model.perturbation_radius
    plot_postprocessed_controls(
        [(t_nom, u_nom, "nominal solution"), (t_rn, u_rn, "SAA solution")],
        os.path.join(run_dir, "postprocessed_controls_%s.png" % stamp),
        N=N, q=q, r=r)
    plot_direct_controls(
        [(td_nom, ud_nom, "nominal solution"), (td_rn, ud_rn, "SAA solution")],
        os.path.join(run_dir, "direct_controls_%s.png" % stamp),
        N=N, q=q, r=r)
