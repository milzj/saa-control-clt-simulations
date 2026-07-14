"""Confidence intervals for the ethanol fermentation SAA optimal value (Ipopt).

Runs the two inference algorithms from ``ensemblecontrol.inference`` on the ethanol
fermentation reactor:

  * plug-in CI (Algorithm 1)     -- normal interval under a unique optimizer,
  * subsampling CI (Algorithm 2) -- interval valid for nonunique optimizers.

Every SAA solve carries the endpoint volume cap x4(t_f) <= 200
(terminal_constraints, member 0). Because x4' = u is parameter-independent, x4(t_f)
is identical across the ensemble, so the cap is position-0 on any (sub)sample and
needs no index remapping. Both the anchor solves and the subsampling re-solves use
Ipopt at the inference tolerance (1e-5) on a 50-interval mesh.

Choose which to run with ``--algorithm {plugin,subsampling,both}`` (default both).
Each algorithm saves its raw data to JSON, then the figures are rendered from those
saved files -- so you can re-plot without re-solving.

Usage (from the repo root, or via scripts/ethanol_fermentation/run_inference.sh):
    MPLBACKEND=Agg python scripts/ethanol_fermentation/inference.py
    ... --algorithm subsampling --m 200 --b 32
"""

import argparse
import os
from datetime import datetime

from casadi import inf

import numpy as np

import ensemblecontrol
from ensemblecontrol import build_lock
from saa_clt.outputs import study_dir
from model import EthanolFermentation
from config import (SAMPLE_SIZES, INFERENCE_SEED, INFERENCE_SUBSAMPLE_SEED,
                    INFERENCE_NINTERVALS, CONFIDENCE_LEVELS, CAP, ipopt_options,
                    TOL_INFERENCE, project_interior, feasible_ramp, substeps_for)

# Study parameters from config (single source of truth). SAMPLE_SIZES is the
# sweep shared with the CLT study; the last entry is the subsampling anchor.
SEED = INFERENCE_SEED                   # scenario-sampler seed
NINTERVALS = INFERENCE_NINTERVALS       # coarse control grid for the repeated SAA solves
SUBSTEPS = substeps_for(NINTERVALS)     # = 32 at 50 intervals (integration density ~1600)

# Every figure is saved in both of these formats.
FORMATS = ("png", "pdf")


def _capped_saa(model, samples):
    # Ipopt single-shooting SAA with the endpoint volume cap on member 0.
    return ensemblecontrol.SAAProblem(
        model, samples, MultipleShooting=False, steps_per_interval=SUBSTEPS,
        terminal_constraints=[(0, lambda x: x[3], -inf, CAP)],
        ipopt_options=ipopt_options(TOL_INFERENCE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("plugin", "subsampling", "both"),
                        default="both", help="which CI algorithm(s) to run")
    parser.add_argument("--m", type=int, default=None,
                        help="number of subsamples (Algorithm 2); "
                             "default 5N per N (growing across the sweep)")
    parser.add_argument("--b", type=int, default=None,
                        help="subsample size (Algorithm 2); default "
                             "floor(N^(6/7)) per sample size; must be < N")
    parser.add_argument("--seed-sub", type=int, default=INFERENCE_SUBSAMPLE_SEED,
                        help="RNG seed for the subsample index draws")
    parser.add_argument("--workers", default="auto",
                        help="parallelism over the m subsampling re-solves per N")
    args = parser.parse_args()

    run_plugin = args.algorithm in ("plugin", "both")
    run_sub = args.algorithm in ("subsampling", "both")

    model = EthanolFermentation()
    model.nintervals = NINTERVALS   # CLT / inference mesh (coarse control grid)
    lb = model.control_bounds[0][0]
    ub = model.control_bounds[1][0]

    def solve(samples, w0=None, inner_serial=False):
        # anchor capped SAA solve via Ipopt (single shooting, tol 1e-5), warm-started
        # from a feasible ramp (the default Ipopt start violates the volume cap).
        saa = _capped_saa(model, samples)
        N = model.nintervals
        ctrl = (project_interior(saa.control_matrix(w0), lb, ub) if w0 is not None
                else feasible_ramp(model, CAP).reshape(N, model.ncontrols))
        saa.initial_decisions = saa.initial_from_controls(ctrl)
        w_opt, f_opt = saa.solve()
        return saa, w_opt, float(np.ravel(f_opt)[0])

    # Ipopt subsampling resolver: subproblem() refuses terminal-constrained problems,
    # so rebuild a fresh capped SAAProblem on each subsample. The cap stays position 0
    # (x4(t_f) is parameter-independent -> equivalent on any subsample), warm-started
    # (interior-clipped) from the full-sample control, under the same build lock the
    # default resolver uses so it is safe under the sweep's outer threads.
    def resolve_for(N, saa, w_opt):
        full = np.asarray(saa.samples, dtype=float)
        controls = project_interior(saa.control_matrix(w_opt), lb, ub)

        def resolve(indices):
            idx = np.asarray(indices, dtype=int)
            with build_lock:
                sub = _capped_saa(model, full[idx])
                sub.initial_decisions = sub.initial_from_controls(controls)
            return float(np.ravel(sub.solve()[1])[0])

        return resolve

    sampler = model.scenario_sampler(seed=SEED)
    samples = sampler.sample(SAMPLE_SIZES[-1])
    N_full = SAMPLE_SIZES[-1]

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = study_dir(__file__, "inference", stamp)

    # Solve each size-N SAA once (nested prefixes). Reused by both algorithms.
    solves = ensemblecontrol.solve_saa_prefixes(solve, samples, SAMPLE_SIZES)

    plugin95 = sub95 = None
    records = sub_records = None
    plugin_path = sub_path = None

    if run_plugin:
        # plug-in CI with the IN-SAMPLE standard deviation (Algorithm 1). The
        # plug-in path is constraint-agnostic: it re-simulates F at the optimizer.
        records = ensemblecontrol.plugin_sweep(solves, SAMPLE_SIZES,
                                               levels=CONFIDENCE_LEVELS)
        plugin_path = os.path.join(run_dir, "plugin.json")
        ensemblecontrol.save_plugin_run(
            records, plugin_path, r=model.perturbation_radius,
            meta={"model": "EthanolFermentation", "sampler": "UniformRelativeSampler",
                  "seed": SEED, "cap": CAP})
        plugin95 = records[-1]["ci"]["levels"][0.95]

    if run_sub:
        b_of = ((lambda N: args.b) if args.b is not None
                else ensemblecontrol.default_subsample_size)
        m = args.m   # None -> per-N default m_N = 5N; an int overrides with a constant
        try:
            sub_records = ensemblecontrol.subsampling_sweep(
                solves, SAMPLE_SIZES, b_of=b_of, m=m, seed=args.seed_sub,
                levels=CONFIDENCE_LEVELS, workers=args.workers,
                resolve_for=resolve_for, progress=True)
        except ValueError as err:
            parser.error(str(err))
        sub_path = os.path.join(run_dir, "subsampling.json")
        ensemblecontrol.save_subsampling_run(
            sub_records, sub_path, r=model.perturbation_radius,
            meta={"resolver": "ipopt-warmstart-capped", "rng_seed": args.seed_sub,
                  "cap": CAP})
        sub95 = sub_records[-1]["ci"]["levels"][0.95]

    # Render the figures (from the saved data). Share the optimal-value y-axis on
    # the _ci{level} interval plots across the plug-in and subsampling families.
    ci_groups = []
    if run_plugin:
        ci_groups.append([r["ci"] for r in records])
    if run_sub:
        ci_groups.append([r["ci"] for r in sub_records])
    value_ylim = None
    if len(ci_groups) >= 2:
        value_ylim = ensemblecontrol.value_ylim_across(
            [ci for group in ci_groups for ci in group])
    if run_plugin:
        ensemblecontrol.plot_plugin(plugin_path, outdir=run_dir, stamp=stamp,
                                    value_ylim=value_ylim, formats=FORMATS)
    if run_sub:
        ensemblecontrol.plot_subsampling(sub_path, outdir=run_dir, stamp=stamp,
                                         value_ylim=value_ylim, formats=FORMATS)

    print("\n95% confidence interval for J_hat_N* (N = {}):".format(N_full))
    if plugin95 is not None:
        print("  plug-in (in-sample s.d.) : [{:.6e}, {:.6e}]  half-width {:.6e}"
              .format(plugin95["lo"], plugin95["hi"], plugin95["halfwidth"]))
    if sub95 is not None:
        print("  subsampling              : [{:.6e}, {:.6e}]  half-width {:.6e}"
              .format(sub95["lo"], sub95["hi"], (sub95["hi"] - sub95["lo"]) / 2.0))


if __name__ == "__main__":
    main()
