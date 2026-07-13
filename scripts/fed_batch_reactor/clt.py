"""Monte-Carlo illustration of the statistical limit theorem for the SAA value.

For the fed-batch reactor, the theory predicts

    N^{1/2}(J_hat_N* - J*)  =>  centered Gaussian     (unique optimizer),

where J_hat_N* is the SAA optimal value on N scenarios and J* is the population
optimal value.  J* is not computable, so we proxy it by J_hat_ref*, the SAA value
on an independent reference sample of size N_ref = 4096.

For each N in {32, 64, 128} we form R replicate SAA optimal values J_hat_N* and
histogram the statistic sqrt(N)*(J_hat_N* - J_hat_ref*).  As N grows the histogram
should look increasingly Gaussian.  The R replicates use COMMON RANDOM NUMBERS across
N (the canonical SAA construction): each replicate draws max(N) scenarios once and its
size-N value uses the first N.

This is a Monte-Carlo study (R solves per N), so it lives in its own script rather than
in the main demos.  It solves with Ipopt at the inference tolerance (1e-5), cold-starting
every solve -- reference and replicates alike -- from the control-box midpoint
u=(lb+ub)/2, so all solves are the same estimator.  The raw replicate values are saved to
results/fed_batch_reactor/limit_theorem/<stamp>/clt.json and the histograms are rendered from them via
``ensemblecontrol.plot_clt``, so the figures can be re-drawn (restyled) without
re-running the ~R*len(NS) solves.
"""

import argparse
import os
from datetime import datetime

import numpy as np

import ensemblecontrol
from ensemblecontrol import build_lock
from saa_clt.outputs import study_dir
from model import FedBatchReactor
from config import (SAMPLE_SIZES, CLT_N_REF, CLT_REPLICATIONS,
                    CLT_ROOT_SEED, CLT_NINTERVALS, ipopt_options,
                    TOL_INFERENCE)

# -- study parameters from config (single source of truth) ---------------
NS = SAMPLE_SIZES      # sample sizes for the statistic (shared CLT + CI sweep)
N_REF = CLT_N_REF      # independent reference sample size (proxies J*); large so
                       # the sqrt(N/N_ref) location drift is negligible
R = CLT_REPLICATIONS   # replicate SAA solves per sample size
ROOT_SEED = CLT_ROOT_SEED   # root entropy; independent child streams are spawned from it

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=R, help="replicate solves per N")
    ap.add_argument("--n-ref", type=int, default=N_REF, help="reference sample size")
    ap.add_argument("--workers", default="auto",
                    help="parallelism over the R replicate solves per N: 'auto' "
                         "(default; no outer threads when a size-N solve already "
                         "saturates the cores), 1, or an integer worker count")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = study_dir(__file__, "limit_theorem", stamp)

    model = FedBatchReactor()
    model.nintervals = CLT_NINTERVALS   # CLT / inference mesh (config)
    # Reproducible i.i.d. scenario root (relative radius r, all params perturbed:
    # xi_j = (1 + r*U[-1,1]) * nominal_j); clt_replication_study splits it into the
    # reference stream plus one group per sample size (each spawning R replicates).
    root = model.scenario_sampler(seed=ROOT_SEED)
    # Cold-start EVERY solve (reference AND replicates) from the SAME fixed guess --
    # the control-box midpoint (average of lower and upper bounds) -- so all solves
    # are the same estimator. This removes the cold(reference)/warm(replicate)
    # asymmetry that made the reference land in a different (lower) basin. Single-
    # shooting Ipopt at the inference tolerance (1e-5); build under the lock, solve
    # outside, so it is safe under the study's threads.
    lb = np.asarray(model.control_bounds[0], dtype=float)
    ub = np.asarray(model.control_bounds[1], dtype=float)
    mid = 0.5 * (lb + ub)   # midpoint control (strictly interior; fed-batch has no cap)

    def solve(samples, w0=None, inner_serial=False):
        kw = dict(parallelization="serial", n_threads=1) if inner_serial else {}
        with build_lock:
            saa = ensemblecontrol.SAAProblem(
                model, samples, MultipleShooting=False,
                ipopt_options=ipopt_options(TOL_INFERENCE), **kw)
            saa.initial_decisions = saa.initial_from_controls(
                np.tile(mid, (model.nintervals, 1)))
        w_opt, f_opt = saa.solve()
        return saa, w_opt, float(np.ravel(f_opt)[0])

    def progress(N, done, total):
        if done % 50 == 0 or done == total:
            print("  N=%3d  replicate %3d/%d" % (N, done, total))

    study = ensemblecontrol.clt_replication_study(
        root, solve, sample_sizes=NS, R=args.R, n_ref=args.n_ref,
        warm_start=False, workers=args.workers, progress=progress)

    print("reference J_hat_ref* (N_ref=%d) = %.8f" % (study["n_ref"], study["f_ref"]))
    for N in NS:
        stat = ensemblecontrol.clt_statistic(study["values_by_N"][N], N,
                                             study["f_ref"])
        print("  N=%3d  mean=%.4f std=%.4f" % (N, stat.mean(), stat.std()))

    # q = mesh size = number of control intervals (= len of the control vector);
    # r = scenario radius (both listed in the plot legends).
    path = ensemblecontrol.save_clt_run(
        study["values_by_N"], os.path.join(run_dir, "clt.json"),
        N_ref=study["n_ref"], f_ref=study["f_ref"], q=study["q"],
        r=model.perturbation_radius,
        meta={"model": "FedBatchReactor", "sampler": "UniformRelativeSampler",
              "sigma": model.perturbation_radius, "seed": ROOT_SEED})
    ensemblecontrol.plot_clt(path, outdir=run_dir, stamp=stamp,
                             formats=("png", "pdf"))   # from saved JSON
    # optimization-bias diagnostic: mean SAA optimal value E[Jhat_N*] + reference
    ensemblecontrol.plot_optimization_bias(path, outdir=run_dir, stamp=stamp,
                                           formats=("png", "pdf"))
    # monotonicity diagnostic of the per-N means (paired/CRN standard errors): is any
    # dip in E[Jhat_N*] a real violation (z < -2) or just finite-R Monte-Carlo noise?
    rows = ensemblecontrol.monotonicity_check(study["values_by_N"], nested=True)
    table = ensemblecontrol.format_monotonicity_table(rows)
    print("\n[clt] monotonicity of E[Jhat_N*] (paired CRN standard errors):")
    print(table)
    with open(os.path.join(run_dir, "monotonicity.txt"), "w") as fh:
        fh.write(table + "\n")
    ensemblecontrol.plot_monotonicity(path, outdir=run_dir, stamp=stamp,
                                      formats=("png", "pdf"))


if __name__ == "__main__":
    main()
