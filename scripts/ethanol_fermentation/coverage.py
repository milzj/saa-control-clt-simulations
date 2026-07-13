"""Monte-Carlo coverage test of the PLUG-IN confidence interval for the ethanol
fermentation SAA optimal value.

A (1 - beta) confidence interval for the SAA optimal value is meant to cover the
population optimal J* with probability 1 - beta.  J* is not computable, so we proxy
it by J_hat_ref*, the SAA value on one large independent reference sample of size
N_ref.  For each N in SAMPLE_SIZES we run R replications -- with COMMON RANDOM NUMBERS
across N (each replicate draws max(N) scenarios once and its size-N CI uses the first
N; the reference sample stays independent) -- build the plug-in CI on each, and count
the L that cover J_hat_ref*.  The empirical coverage is L/R; the estimator
``probability_lower_bound(R, L, delta)`` upgrades it to a rigorous (1 - delta) lower
confidence bound on the true coverage (Clopper-Pearson; eq. 10.2.4 / Lemma 10.2.1).

Every SAA solve carries the endpoint volume cap x4(t_f) <= CAP (terminal_constraints,
member 0).  Because x4' = u is parameter-independent, x4(t_f) is identical across the
ensemble, so the cap is position 0 on any (sub)sample and needs no index remapping.
Solves cold-start from a FEASIBLE ramp (the default Ipopt midpoint start violates the
cap) at the inference tolerance -- the same estimator as the CLT / inference drivers.

The plug-in CI (Algorithm 1) costs exactly R + 1 SAA solves per N -- one extra rollout
per replication, no re-solve.  The R replicate solves run in parallel (default cpu-2)
while each solve stays serial.  Raw indicators are saved to
results/ethanol_fermentation/coverage/<stamp>/coverage_plugin.json and the LaTeX table
to coverage_plugin.tex, so the table can be re-derived via
``ensemblecontrol.coverage_latex_table`` without re-running.

Usage (from the repo root; MPLBACKEND=Agg on headless machines):
    python scripts/ethanol_fermentation/coverage.py
    ... --R 100 --n-ref 512          # cheaper smoke run
"""

import argparse
import os
from datetime import datetime

from casadi import inf

import numpy as np

import ensemblecontrol
from ensemblecontrol import build_lock, core_budget   # CasADi build lock; cpu-2
from saa_clt.outputs import study_dir
from model import EthanolFermentation
from config import (SAMPLE_SIZES, COVERAGE_ROOT_SEED, COVERAGE_REPLICATIONS,
                    COVERAGE_N_REF, COVERAGE_NINTERVALS, COVERAGE_LEVELS, CAP,
                    ipopt_options, TOL_INFERENCE, feasible_ramp, substeps_for)

SUBSTEPS = substeps_for(COVERAGE_NINTERVALS)   # RK4 substeps to hold ~1600-step density


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", type=int, default=COVERAGE_REPLICATIONS,
                    help="plug-in replications per N (1 solve each; default %d)"
                         % COVERAGE_REPLICATIONS)
    ap.add_argument("--n-ref", type=int, default=COVERAGE_N_REF,
                    help="reference sample size proxying J* (default %d)" % COVERAGE_N_REF)
    ap.add_argument("--deltas", default="1e-6",
                    help="comma-separated failure probabilities for the lower bound "
                         "(delta=1e-6 -> (1 - 1e-6)-confident lower bound); the first "
                         "value also sets the .txt summary's bound")
    ap.add_argument("--workers", default=str(core_budget()),
                    help="outer parallelism over the R replicate solves per N: default "
                         "cpu-2 (= %d here) runs that many solves at once, each serial; "
                         "pass an integer, 1 (sequential inner-threaded), or 'auto'."
                         % core_budget())
    args = ap.parse_args()
    args.deltas = tuple(float(d) for d in args.deltas.split(","))

    # One timestamped run folder per study: output/coverage/<stamp>/ (matches the
    # CLT / inference / mean_saa studies), so re-runs never overwrite each other.
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = study_dir(__file__, "coverage", stamp)

    model = EthanolFermentation()
    model.nintervals = COVERAGE_NINTERVALS   # coarse control mesh (matches CLT / inference)
    cold = feasible_ramp(model, CAP).reshape(model.nintervals, model.ncontrols)

    # Ipopt single-shooting capped solve at the inference tolerance, silenced (many
    # solves), cold-started from the feasible ramp -- same estimator as the CLT /
    # inference drivers (w0 unused, so warm_start below only fixes the reference target).
    quiet = dict(ipopt_options(TOL_INFERENCE), print_level=0, sb="yes")

    def solve(samples, w0=None, inner_serial=False):
        kw = dict(parallelization="serial", n_threads=1) if inner_serial else {}
        with build_lock:
            saa = ensemblecontrol.SAAProblem(
                model, samples, MultipleShooting=False, steps_per_interval=SUBSTEPS,
                terminal_constraints=[(0, lambda x: x[3], -inf, CAP)],
                ipopt_options=quiet, **kw)
            saa.initial_decisions = saa.initial_from_controls(cold)
        w_opt, f_opt = saa.solve()
        return saa, w_opt, float(np.ravel(f_opt)[0])

    root = model.scenario_sampler(seed=COVERAGE_ROOT_SEED)

    def progress(N, done, total):
        # First replication and every ~10% (plus the last) so each N block shows a
        # prompt start marker, then coarse progress, without flooding the shell.
        if done == 1 or done % max(1, total // 10) == 0 or done == total:
            print("  [plugin] N=%3d  replication %5d/%d" % (N, done, total), flush=True)

    def ref_solve(samples):
        # The reference SAA (N_ref -- the single largest solve) runs before any
        # replication; announce it so the shell is not silent through that whole solve.
        print("[plugin] solving reference SAA (N_ref=%d)..." % args.n_ref, flush=True)
        out = solve(samples)
        print("[plugin] reference J_hat_ref* (N_ref=%d) = %.8f"
              % (args.n_ref, out[2]), flush=True)
        return out

    print("[plugin] coverage study: sizes=%s, R=%d, workers=%s (%d replicate solves)"
          % (list(SAMPLE_SIZES), args.R, args.workers, args.R * len(SAMPLE_SIZES)),
          flush=True)

    # ci_of=None -> the default plug-in CI (one extra rollout, no re-solve).
    study = ensemblecontrol.coverage_study(
        root, solve, sample_sizes=SAMPLE_SIZES, R=args.R, n_ref=args.n_ref,
        levels=COVERAGE_LEVELS, warm_start=False, workers=args.workers,
        progress=progress, ref_solve=ref_solve)

    path = ensemblecontrol.save_coverage_run(
        study, os.path.join(run_dir, "coverage_plugin.json"),
        meta={"model": "EthanolFermentation", "ci": "plugin",
              "sampler": "UniformRelativeSampler",
              "radius": model.perturbation_radius, "seed": COVERAGE_ROOT_SEED,
              "cap": CAP},
        delta=args.deltas[0])   # .txt summary uses the first delta (matches the .tex)

    caption = ("Estimated coverage of the plug-in confidence interval (Algorithm 1) "
               "for the SAA optimal value of the ethanol fermentation reactor. The "
               "population optimum is proxied by the SAA optimal value on an independent "
               "reference sample of size $N_{\\mathrm{ref}} = %d$. For each training "
               "size $N$ and nominal level $1-\\beta$, $L/R$ is the empirical coverage "
               "over $R=%d$ replications and $\\widehat p_{R,\\delta}(L)$ is the "
               "$(1-\\delta)$ lower confidence bound on the true coverage."
               % (study["n_ref"], study["R"]))
    tex = ensemblecontrol.coverage_latex_table(
        path, deltas=args.deltas, caption=caption, label="tab:coverage_plugin")
    with open(os.path.join(run_dir, "coverage_plugin.tex"), "w") as fh:
        fh.write(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()
