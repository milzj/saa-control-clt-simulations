"""Nominal and risk-neutral (SAA) solutions for the ethanol fermentation reactor (Ipopt).

Solves the nominal problem and the risk-neutral SAA problem on a fine 2000-interval
mesh with the endpoint volume cap x4(t_f) <= 200 enforced via
SAAProblem(terminal_constraints=...), plots the state and control trajectories, and
reconstructs the singular control by phi-postprocessing (see saa_clt.postprocess).

Because x4' = u is parameter-independent, x4(t_f) is identical across the ensemble,
so the cap is enforced on a single member and is effectively a control-space budget.
The objective J = x3(t_f)*x4(t_f) is very flat near constant feed, so the solve is
warm-started from a feasible rising ramp that sits just inside the cap budget.

Outputs go to results/ethanol_fermentation/controls-states/<stamp>/{nominal,saa}/
(per-solution control and state plots) with the nominal-vs-SAA overlays at the
run-folder top level; every figure is saved as both PNG and PDF.

Run from the repo root (or via scripts/ethanol_fermentation/run_nominal_saa.sh):
    MPLBACKEND=Agg python scripts/ethanol_fermentation/nominal_saa.py
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from casadi import inf

import ensemblecontrol
from saa_clt.outputs import study_dir
from saa_clt.postprocess import (plot_phi_constructed_control,
                                 plot_postprocessed_controls,
                                 plot_direct_controls, annotate_nominal)
from model import EthanolFermentation
from config import (SOLUTION_NINTERVALS, SAA_SAMPLES, SAA_SAMPLER_SEED, CAP,
                    ipopt_options, TOL_SOLUTION, feasible_ramp, substeps_for)

# Fine mesh for the nominal / risk-neutral solutions (config).
NINTERVALS = SOLUTION_NINTERVALS

# RK4 substeps per control interval for the SOLVE, sized to the mesh so the total
# integration density stays ~constant (see ipopt_options.substeps_for). The stiff
# ~61 h dynamics need enough INTEGRATION steps or Ipopt games the RK4 error in
# J = x3(t_f)*x4(t_f); at this fine 2000-interval grid the control grid alone already
# suffices (substeps_for(2000) = 1), whereas the coarse 50-interval CLT/inference
# grid needs 32.
SUBSTEPS = substeps_for(NINTERVALS)

# phi postprocessing grid: number of sub-intervals per direct-solve interval.
POSTPROC_S = 1

# Every figure is saved in both of these formats.
FORMATS = ("png", "pdf")

# CAP (endpoint volume constraint x4(t_f) <= CAP) comes from config, imported
# above. The risk-neutral ensemble is always i.i.d. Monte Carlo.


def solve(model, samples, w0=None):
    # Ipopt single-shooting SAA solve with the endpoint volume cap on member 0
    # (x4 is state index 3); x4(t_f) is parameter-independent so this caps the whole
    # ensemble. w0 (else a feasible ramp) warm-starts the interior-point solve.
    N = model.nintervals
    nu = model.ncontrols
    h = model.final_time / N
    x40 = float(model.parameterized_initial_state(
        np.atleast_2d(np.asarray(samples, float))[0])[3])

    saa = ensemblecontrol.SAAProblem(
        model, samples, MultipleShooting=False, steps_per_interval=SUBSTEPS,
        terminal_constraints=[(0, lambda x: x[3], -inf, CAP)],
        ipopt_options=ipopt_options(TOL_SOLUTION))

    if w0 is None:
        w0 = feasible_ramp(model, CAP)
    saa.initial_decisions = saa.initial_from_controls(
        np.asarray(w0, float).reshape(N, nu))

    w_opt, f_opt = saa.solve()
    J = -float(np.ravel(f_opt)[0])
    x4_end = x40 + h * float(np.sum(np.asarray(w_opt).ravel()[:N * nu]))
    print("[saa] J = x3(tf)*x4(tf) = %.1f   x4(tf) = %.2f L" % (J, x4_end))
    return saa, w_opt


def _save_both(fig, base_no_ext):
    for fmt in FORMATS:
        fig.savefig("%s.%s" % (base_no_ext, fmt), bbox_inches="tight")


def _plot_states(plotter, case, base, nsigma, deterministic=False):
    # per-state trajectories with a +/- nsigma ensemble band; clean "<case> x{j}"
    # legends and "<base>_x{j}.{png,pdf}" filenames.
    radius = plotter.saa_problem.control_problem.perturbation_radius
    figs, axes = plotter.plot_states(
        per_state=True, radius=None if deterministic else radius,
        annotate=not deterministic, nsigma=nsigma,
        state_labels=["%s x%d" % (case, j) for j in range(plotter.nstates)])
    if deterministic:
        # q only: the nominal solve has no ensemble and no perturbation.
        for axis in np.atleast_1d(axes):
            annotate_nominal(axis, plotter.nintervals)
    for j, fig in enumerate(figs):
        _save_both(fig, "%s_x%d" % (base, j))
        plt.close(fig)


def plot(saa_problem, w_opt, file_prefix, case, case_dir, stamp, nsigma=1,
         deterministic=False):
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    radius = saa_problem.control_problem.perturbation_radius
    base = os.path.join(case_dir, "%s_%s" % (file_prefix, stamp))

    if deterministic:
        # SolutionPlotter's annotation always carries N (and r when a radius is
        # given); the nominal solve has neither, so annotate with q ourselves.
        fig, ax = plotter.plot_controls(control_labels=["%s control" % case],
                                        step=True, annotate=False)
        annotate_nominal(ax, plotter.nintervals)
        _save_both(fig, base + "_control")
    else:
        plotter.plot_controls(control_labels=["%s control" % case], step=True,
                              radius=radius, savepath=base + "_control.png",
                              formats=FORMATS)
    plt.close("all")
    _plot_states(plotter, case, base, nsigma, deterministic=deterministic)

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
    # results/ethanol_fermentation/controls-states/<stamp>/{nominal,saa}/
    run_dir = study_dir(__file__, "controls-states", stamp)
    NOMINAL_DIR = os.path.join(run_dir, "nominal")
    SAA_DIR = os.path.join(run_dir, "saa")
    for _d in (NOMINAL_DIR, SAA_DIR):
        os.makedirs(_d, exist_ok=True)

    # -- Nominal problem -----------------------------------------------------
    nominal_model = EthanolFermentation()
    nominal_model.nintervals = NINTERVALS
    nominal_param = nominal_model.nominal_param
    saa_nom, w_nom = solve(nominal_model, nominal_param)
    td_nom, ud_nom = plot(saa_nom, w_nom, "nominal", "nominal", NOMINAL_DIR,
                          stamp, deterministic=True)
    # constructed control via nominal phi_j (switching structure from this run)
    t_nom, u_nom = plot_phi_constructed_control(
        nominal_model, nominal_param, w_nom, "nominal", outdir=NOMINAL_DIR,
        S=POSTPROC_S, stamp=stamp, case="nominal", deterministic=True)

    # -- Risk-neutral (SAA) problem ------------------------------------------
    saa_model = EthanolFermentation()
    saa_model.nintervals = NINTERVALS

    # sampler: relative multiplicative perturbation of the 5 kinetic constants
    # (k5, k6 frozen), radius r from the model (EthanolFermentation.scenario_sampler).
    N = SAA_SAMPLES
    samples = saa_model.scenario_sampler(seed=SAA_SAMPLER_SEED).sample(N)

    # warm-start the SAA from the (feasible) nominal control
    saa_rn, w_rn = solve(saa_model, samples, w0=ud_nom)
    td_rn, ud_rn = plot(saa_rn, w_rn, "saa", "SAA", SAA_DIR, stamp, nsigma=3)
    # constructed control via ensemble phi_j (switching structure from this run)
    t_rn, u_rn = plot_phi_constructed_control(
        saa_model, samples, w_rn, "saa", outdir=SAA_DIR, S=POSTPROC_S, stamp=stamp,
        case="SAA")
    # states obtained by rolling the postprocessed (phi) control through the dynamics
    plot_postprocessed_states(saa_rn, u_rn, "SAA postprocessed", SAA_DIR, stamp,
                              nsigma=3)

    # -- Nominal vs SAA overlays (at the run-folder top level) ----------------
    # One overlay of the direct (solved) controls, one of their postprocessed (phi)
    # counterparts -- written to the timestamped run folder (like the fed-batch demo).
    q = saa_model.nintervals
    r = saa_model.perturbation_radius
    direct_cases = [(td_nom, ud_nom, "nominal control"),
                    (td_rn, ud_rn, "SAA control")]
    post_cases = [(t_nom, u_nom, "postprocessed nominal control"),
                  (t_rn, u_rn, "postprocessed SAA control")]
    plot_postprocessed_controls(
        post_cases, os.path.join(run_dir, "postprocessed_controls_%s.png" % stamp),
        N=N, q=q, r=r)
    plot_direct_controls(
        direct_cases, os.path.join(run_dir, "direct_controls_%s.png" % stamp),
        N=N, q=q, r=r)
