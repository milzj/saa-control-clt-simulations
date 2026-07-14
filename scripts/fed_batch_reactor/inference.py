"""Confidence intervals for the fed-batch reactor SAA optimal value (Ipopt).

Runs the two inference algorithms from ``ensemblecontrol.inference`` on the
fed-batch reactor:

  * plug-in CI (Algorithm 1)     -- normal interval under a unique optimizer,
  * subsampling CI (Algorithm 2) -- interval valid for nonunique optimizers.

Both the anchor SAA solves and the subsampling re-solves use Ipopt at the
inference tolerance (1e-5) on a 50-interval mesh.  Choose which to run with
``--algorithm {plugin,subsampling,both}`` (default both).  The full-sample
(size-N) SAA is solved once and reused: it is the largest plug-in sample size
AND the anchor J_hat_N* / u_hat_N for subsampling.

Each algorithm saves its raw data to JSON, then the figures are rendered from
those saved files -- so you can re-plot (rename labels, change CI bands) without
re-solving; see the plot-only path below.

Usage (from the repo root, or via scripts/fed_batch_reactor/run_inference.sh):
    MPLBACKEND=Agg python scripts/fed_batch_reactor/inference.py
    ... --algorithm plugin
    ... --algorithm subsampling --m 200 --b 32
Re-plot only, no solves:
    python -c "import ensemblecontrol as e; \
        e.plot_plugin('results/fed_batch_reactor/inference/<stamp>/plugin.json', outdir='.')"
"""

import argparse
import os
from datetime import datetime

import ensemblecontrol
from ensemblecontrol import build_lock
from saa_clt.outputs import study_dir
from model import FedBatchReactor
from config import (SAMPLE_SIZES, INFERENCE_SEED, INFERENCE_SUBSAMPLE_SEED,
                    INFERENCE_NINTERVALS, CONFIDENCE_LEVELS, ipopt_options,
                    TOL_INFERENCE, project_interior)

# Study parameters from config. SAMPLE_SIZES is the sweep shared with the CLT
# study; the last entry is the subsampling anchor.
SEED = INFERENCE_SEED          # scenario-sampler seed

# Every figure is saved in both of these formats.
FORMATS = ("png", "pdf")


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
                        help="parallelism over the m subsampling re-solves per N: "
                             "'auto' (default; no outer threads when a size-b solve "
                             "already saturates the cores), 1, or an integer count")
    args = parser.parse_args()

    run_plugin = args.algorithm in ("plugin", "both")
    run_sub = args.algorithm in ("subsampling", "both")

    model = FedBatchReactor()
    model.nintervals = INFERENCE_NINTERVALS   # CLT / inference mesh (config)
    lb = model.control_bounds[0][0]
    ub = model.control_bounds[1][0]

    def solve(samples, w0=None, inner_serial=False):
        # anchor SAA solve via Ipopt (single shooting, tol 1e-5); the studies'
        # solve(samples) -> (saa, w_opt, f_opt) contract. The anchor sweep is
        # sequential, so w0/inner_serial are unused here.
        saa = ensemblecontrol.SAAProblem(
            model, samples, MultipleShooting=False,
            ipopt_options=ipopt_options(TOL_INFERENCE))
        w_opt, f_opt = saa.solve()
        return saa, w_opt, f_opt

    # Ipopt subsampling resolver: re-solve each size-b subsample with Ipopt (the
    # subproblem inherits the anchor's ipopt_options, i.e. tol 1e-5), warm-started
    # from the full-sample control clipped strictly interior for the interior-point
    # method. Construction is serialized with the same build lock the default
    # (scipy) resolver uses, so it is safe under the sweep's outer threads.
    def resolve_for(N, saa, w_opt):
        controls = project_interior(saa.control_matrix(w_opt), lb, ub)

        def resolve(indices):
            with build_lock:
                sub = saa.subproblem(indices)
                sub.initial_decisions = sub.initial_from_controls(controls)
            return sub.solve()[1]

        return resolve

    sampler = FedBatchReactor().scenario_sampler(seed=SEED)
    samples = sampler.sample(SAMPLE_SIZES[-1])
    N_full = SAMPLE_SIZES[-1]

    # One timestamped inference folder per run:
    # results/fed_batch_reactor/inference/<stamp>/.
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = study_dir(__file__, "inference", stamp)

    # Solve each size-N SAA once (nested prefixes of the scenario draw). Reused by
    # BOTH algorithms: the plug-in sweep and, at each N, the subsampling anchor
    # J_hat_N* / u_hat_N -- so subsampling adds only the m subsample re-solves per
    # N, no extra size-N solves.
    solves = ensemblecontrol.solve_saa_prefixes(solve, samples, SAMPLE_SIZES)

    plugin95 = sub95 = None
    records = sub_records = None
    plugin_path = sub_path = None

    if run_plugin:
        # plug-in CI with the IN-SAMPLE standard deviation (Algorithm 1)
        records = ensemblecontrol.plugin_sweep(solves, SAMPLE_SIZES,
                                               levels=CONFIDENCE_LEVELS)
        plugin_path = os.path.join(run_dir, "plugin.json")
        ensemblecontrol.save_plugin_run(
            records, plugin_path, r=model.perturbation_radius,
            meta={"model": "FedBatchReactor", "sampler": "UniformRelativeSampler",
                  "seed": SEED})
        plugin95 = records[-1]["ci"]["levels"][0.95]

    if run_sub:
        # subsampling at each sample size N -- the analogue of the plug-in sweep.
        # Default block size b = floor(N^{6/7}) PER N (grows with N, b/N -> 0);
        # m_N = 5N subsamples PER N (growing). --b/--m override with fixed values.
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
            meta={"resolver": "ipopt-warmstart", "rng_seed": args.seed_sub})
        sub95 = sub_records[-1]["ci"]["levels"][0.95]

    # Render the figures (from the saved data). Share the optimal-value y-axis on
    # the _ci{level} interval plots across the plug-in and subsampling families for
    # comparison.
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
