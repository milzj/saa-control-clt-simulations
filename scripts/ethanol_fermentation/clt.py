"""Monte-Carlo illustration of the statistical limit theorem for the SAA value.

For the ethanol fermentation reactor, the theory predicts

    N^{1/2}(J_hat_N* - J*)  =>  centered Gaussian     (unique optimizer),

where J_hat_N* is the SAA optimal value on N scenarios and J* is the population
optimal value.  J* is not computable, so we proxy it by J_hat_ref*, the SAA value
on an independent reference sample of size N_ref = 4096.

For each N in SAMPLE_SIZES (config; (8, 16, 32, 64)) we form R replicate SAA optimal
values J_hat_N* and histogram the statistic sqrt(N)*(J_hat_N* - J_hat_ref*).  As N grows
the histogram should look increasingly Gaussian.  The R replicates use COMMON RANDOM
NUMBERS across N (the canonical SAA construction): each replicate draws max(N) scenarios
once and its size-N value uses the first N.

Each SAA solve carries the endpoint volume cap x4(t_f) <= 200
(terminal_constraints, member 0; x4(t_f) is parameter-independent so this caps the
whole ensemble).  The solves use Ipopt at the inference tolerance (TOL_INFERENCE = 1e-3)
and warm-start every replicate from the reference solution.  The raw replicate values
are saved to results/ethanol_fermentation/limit_theorem/<stamp>/clt.json and the
histograms are rendered from them via ``saa_clt.plot_clt_normal``.

Each N gets three histograms -- the fitted normal and the theory curve
N(0, sigma_hat_{N_ref}^2) overlaid, then each curve alone -- plus a combined all-N
overview.  sigma_hat_{N_ref} is the plug-in estimate at the reference solve (the CLT's
asymptotic sigma^2 = Var F(x*, xi)), captured during the run because
``clt_replication_study`` keeps only the scalar optimal value.  The reference solve
itself (control, value, sigma) is written alongside as clt_reference.json.
"""

import argparse
import os
from datetime import datetime

import numpy as np
from casadi import inf

import ensemblecontrol
from ensemblecontrol import build_lock
from saa_clt.outputs import study_dir, save_clt_reference
from saa_clt.clt_plots import plot_clt_normal
from model import EthanolFermentation
from config import (SAMPLE_SIZES, CLT_N_REF, CLT_REPLICATIONS,
                    CLT_ROOT_SEED, CLT_NINTERVALS, CAP, ipopt_options,
                    TOL_INFERENCE, project_interior, feasible_ramp,
                    substeps_for)

# -- study parameters --------------------------------------------------------
# Study parameters from config (single source of truth).
NS = SAMPLE_SIZES              # sample sizes for the statistic (shared CLT + CI sweep)
N_REF = CLT_N_REF              # independent reference sample size (proxies J*)
R = CLT_REPLICATIONS           # replicate SAA solves per sample size
ROOT_SEED = CLT_ROOT_SEED      # root entropy; independent child streams are spawned from it
NINTERVALS = CLT_NINTERVALS    # coarse control grid for the repeated SAA solves
SUBSTEPS = substeps_for(NINTERVALS)   # = 32 at 50 intervals (integration density ~1600)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=R, help="replicate solves per N")
    ap.add_argument("--n-ref", type=int, default=N_REF, help="reference sample size")
    ap.add_argument("--workers", default="auto",
                    help="parallelism over the R replicate solves per N: 'auto' "
                         "(default), 1, or an integer worker count")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = study_dir(__file__, "limit_theorem", stamp)

    model = EthanolFermentation()
    model.nintervals = NINTERVALS   # CLT / inference mesh (coarse control grid)
    # Reproducible i.i.d. scenario root (relative radius r; k5, k6 frozen).
    root = model.scenario_sampler(seed=ROOT_SEED)
    # Ipopt single-shooting solve at the inference tolerance (TOL_INFERENCE = 1e-3) with
    # the endpoint volume cap. Like make_ipopt_solve but cold-starts (the reference) from a
    # feasible ramp -- the default Ipopt start violates the cap -- and warm-starts each
    # replicate (interior-clipped) from the reference control. Construction is
    # serialized with the build lock so it is safe under the study's outer threads.
    lb = model.control_bounds[0][0]
    ub = model.control_bounds[1][0]

    # Plug-in sigma at the REFERENCE solve, for the CLT plot's N(0, sigma^2) overlay.
    # The limit theorem's variance is sigma^2 = Var F(x*, xi); the reference optimizer
    # (N_ref scenarios) is the best available proxy for x*, so sigma_hat there estimates
    # it.  Captured inside this closure because clt_replication_study keeps only the
    # scalar value and drops the (saa, w_opt) pair terminal_losses needs.
    sigma_ref = {}

    def solve(samples, w0=None, inner_serial=False):
        kw = dict(parallelization="serial", n_threads=1) if inner_serial else {}
        with build_lock:
            saa = ensemblecontrol.SAAProblem(
                model, samples, MultipleShooting=False, steps_per_interval=SUBSTEPS,
                terminal_constraints=[(0, lambda x: x[3], -inf, CAP)],
                ipopt_options=ipopt_options(TOL_INFERENCE), **kw)
            ctrl = (project_interior(saa.control_matrix(w0), lb, ub) if w0 is not None
                    else feasible_ramp(model, CAP).reshape(model.nintervals,
                                                           model.ncontrols))
            saa.initial_decisions = saa.initial_from_controls(ctrl)
        w_opt, f_opt = saa.solve()
        f = float(np.ravel(f_opt)[0])
        # The reference solve: the FIRST solve of size n_ref.  First-match-wins so a
        # replicate cannot clobber it should a small --n-ref collide with a sweep
        # size (n_ref=4096 does not, but smoke runs pass smaller values).
        if saa.nsamples == args.n_ref and "sigma" not in sigma_ref:
            losses = ensemblecontrol.terminal_losses(saa, w_opt)
            sigma_ref["sigma"] = ensemblecontrol.plugin_ci_from_losses(
                losses, f_opt=f)["sigma"]   # same estimator as Algorithm 1's CI
        return saa, w_opt, f

    def progress(N, done, total):
        if done % 50 == 0 or done == total:
            print("  N=%3d  replicate %3d/%d" % (N, done, total))

    study = ensemblecontrol.clt_replication_study(
        root, solve, sample_sizes=NS, R=args.R, n_ref=args.n_ref,
        workers=args.workers, progress=progress)

    print("reference J_hat_ref* (N_ref=%d) = %.8f" % (study["n_ref"], study["f_ref"]))
    print("reference sigma_hat (N_ref=%d) = %.8f   [N(0, sigma^2) overlay]"
          % (study["n_ref"], sigma_ref["sigma"]))
    for N in NS:
        stat = ensemblecontrol.clt_statistic(study["values_by_N"][N], N,
                                             study["f_ref"])
        print("  N=%3d  mean=%.4f std=%.4f" % (N, stat.mean(), stat.std()))

    # q = mesh size = number of control intervals; r = scenario radius.
    path = ensemblecontrol.save_clt_run(
        study["values_by_N"], os.path.join(run_dir, "clt.json"),
        N_ref=study["n_ref"], f_ref=study["f_ref"], q=study["q"],
        r=model.perturbation_radius,
        # NB: meta["sigma"] is the scenario radius (historical name); the variance
        # estimator for the CLT overlay is meta["sigma_ref"].
        meta={"model": "EthanolFermentation", "sampler": "UniformRelativeSampler",
              "sigma": model.perturbation_radius, "seed": ROOT_SEED, "cap": CAP,
              "sigma_ref": sigma_ref["sigma"]})
    # Companion to clt.json: the reference solve itself (control + value + sigma),
    # which save_clt_run drops -- it keeps only the value f_ref.
    save_clt_reference(
        study, os.path.join(run_dir, "clt_reference.json"),
        sigma_ref=sigma_ref["sigma"],
        meta={"model": "EthanolFermentation", "sampler": "UniformRelativeSampler",
              "seed": ROOT_SEED, "cap": CAP, "R": args.R})
    print("[clt] wrote the reference solution -> clt_reference.json")
    plot_clt_normal(path, outdir=run_dir, stamp=stamp,
                    formats=("png", "pdf"))   # from saved JSON
    # optimization-bias diagnostic: mean SAA optimal value E[Jhat_N*] + reference
    ensemblecontrol.plot_optimization_bias(path, outdir=run_dir, stamp=stamp,
                                           formats=("png", "pdf"))


if __name__ == "__main__":
    main()
