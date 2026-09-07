"""Mean subsampling-CI width conditional on one fed-batch dataset.

This paper-revision study draws the inference scenario sample once, solves and
checkpoints one full-sample SAA anchor for each N = 32 and 64, and then repeats
only the random subsample construction and its optimization solves.  The two
saved figures plot the mean interval width over the subsampling repetitions.

This is not a coverage experiment: no population optimum or coverage indicator
is computed.  Within each ``(replication, N)`` block, the subsets for different
``b_N`` are nested prefixes, so comparisons across ``b_N`` are paired.  Every
replication uses the same fixed scenario sample and the same checkpointed
full-sample optimum.

The expensive unit of work is one ``(replication, N)`` block.  Each completed
block is checkpointed in the aggregate JSON file, and ``--resume`` skips all
valid completed blocks.  Resume also reuses saved anchors; it never re-solves a
valid checkpointed full-sample SAA.

Run from the repository root via::

    scripts/fed_batch_reactor/run_subsampling_width_monte_carlo.sh

The default destination is ``results/revision``.  If its completed checkpoint
is present, the command validates and reuses it before recreating the summary
and figures.  Pass ``--outdir`` to perform a fresh run elsewhere.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

import ensemblecontrol
from ensemblecontrol import build_lock, core_budget

from saa_clt.outputs import repo_root
from saa_clt.subsampling_width_mc import (
    ALGORITHM,
    deltas_for_block_size,
    load_width_run,
    nested_index_sets,
    plot_mean_subsampling_width,
    save_width_run,
    save_width_summary,
    subsample_size_grid,
    subsampling_interval_from_deltas,
)
from model import FedBatchReactor
from config import (
    INFERENCE_NINTERVALS,
    INFERENCE_SEED,
    SAMPLE_SIZES,
    SUBSAMPLING_WIDTH_ROOT_SEED,
    TOL_INFERENCE,
    ipopt_options,
    project_interior,
)


DEFAULT_SAMPLE_SIZES = (32, 64)
DEFAULT_REPLICATIONS = 30
DEFAULT_WORKERS = min(8, core_budget())
WIDTH_GRID_PARENT_SIZE = 7
WIDTH_GRID_STRIDE = 2
FORMATS = ("png", "pdf")
RUN_JSON = "subsampling_width_monte_carlo.json"
SUMMARY_CSV = "subsampling_width_monte_carlo_summary95.csv"
LEVEL = 0.95


# Spawned processes cannot safely receive CasADi ``SAAProblem`` objects.  Each
# worker therefore rebuilds one read-only full-sample problem from the fixed
# seed in its initializer, but never solves it.  Tasks carry only scalar ids and
# seeds; anchors are small JSON-compatible values passed once at pool startup.
_WORKER_SAA = None
_WORKER_ANCHORS: dict[str, dict] = {}
_WORKER_SAMPLE_SHA256: str | None = None


def _width_grid(N: int) -> list[int]:
    """Retain alternating points and both endpoints of the seven-point grid."""
    return subsample_size_grid(N, point_count=WIDTH_GRID_PARENT_SIZE)[
        ::WIDTH_GRID_STRIDE
    ]


def _scalar(value) -> float:
    scalar = float(np.asarray(value).squeeze())
    if not math.isfinite(scalar):
        raise ValueError("optimal value must be finite")
    return scalar


def _seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _seed_plan(replications: int, root_seed: int) -> list[dict]:
    """Return independent subset-index streams for every ``(replication,N)``."""
    repetition_streams = np.random.SeedSequence(root_seed).spawn(replications)
    plan = []
    for replication, stream in enumerate(repetition_streams):
        children = stream.spawn(len(SAMPLE_SIZES))
        plan.append({
            "replication": replication,
            "subsample_seed_by_N": {
                str(N): _seed(child)
                for N, child in zip(SAMPLE_SIZES, children)
            },
        })
    return plan


def _fixed_samples(max_sample_size: int) -> np.ndarray:
    sampler = FedBatchReactor().scenario_sampler(seed=int(INFERENCE_SEED))
    samples = np.asarray(sampler.sample(int(max_sample_size)), dtype=float)
    if samples.ndim != 2 or samples.shape[0] != int(max_sample_size):
        raise ValueError("fixed scenario sampler returned an unexpected shape")
    if not np.all(np.isfinite(samples)):
        raise ValueError("fixed scenario sample contains nonfinite values")
    return samples


def _sample_sha256(samples: np.ndarray) -> str:
    canonical = np.ascontiguousarray(samples, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _new_run(
    replications: int,
    sample_sizes: tuple[int, ...],
    root_seed: int,
    samples: np.ndarray,
) -> dict:
    sample_digest = _sample_sha256(samples)
    return {
        "algorithm": ALGORITHM,
        "version": 3,
        "R": int(replications),
        "sample_sizes": list(sample_sizes),
        "level": LEVEL,
        "q": int(INFERENCE_NINTERVALS),
        "r": float(FedBatchReactor().perturbation_radius),
        "root_seed": int(root_seed),
        "fixed_sample": {
            "scenario_seed": int(INFERENCE_SEED),
            "size": int(samples.shape[0]),
            "shape": list(samples.shape),
            "sha256": sample_digest,
        },
        "grid": {
            "point_count": 4,
            "parent_point_count": WIDTH_GRID_PARENT_SIZE,
            "retained_parent_indices": [1, 3, 5, 7],
            "B_N": "floor(N^(6/7))",
            "d_N": "max(1, floor(B_N/7))",
            "b_N_k": "B_N - (6-2k)*d_N, k=0,...,3; retain 1 <= b < N",
            "values_by_N": {
                str(N): _width_grid(N) for N in sample_sizes
            },
        },
        "meta": {
            "model": "FedBatchReactor",
            "sampler": "UniformRelativeSampler",
            "m_expr": "5N",
            "scenario_coupling_across_N": (
                "one fixed size-max(sample_sizes) inference-seed draw; size-N prefixes"
            ),
            "subsample_coupling_across_b": (
                "nested prefixes of ordered size-B_N draws within each (replication,N)"
            ),
            "subsample_streams": (
                "independent SeedSequence child stream for each (replication,N)"
            ),
            "resolver": "ipopt-warmstart from the matching full-sample control",
            "solver_tolerance": float(TOL_INFERENCE),
            "design": "conditional_on_one_fixed_full_sample",
            "full_sample_reused_across_repetitions": True,
            "subsampling_repetitions_independent": True,
            "outer_replications_independent": False,
            "interpretation": (
                "mean width averages only finite-m_N subset-index randomness, "
                "conditional on the fixed full sample; this is not a coverage study"
            ),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        "seed_plan": _seed_plan(replications, root_seed),
        "anchors": {},
        "complete": False,
        "replications": [],
    }


def _anchor_is_valid(anchor: dict, N: int, sample_sha256: str) -> bool:
    try:
        controls = np.asarray(anchor["projected_controls"], dtype=float)
        center = _scalar(anchor["f_opt"])
        expected_controls = (int(INFERENCE_NINTERVALS),
                             int(FedBatchReactor().ncontrols))
        return (
            int(anchor["N"]) == int(N)
            and int(anchor["scenario_seed"]) == int(INFERENCE_SEED)
            and anchor["fixed_sample_sha256"] == sample_sha256
            and controls.shape == expected_controls
            and np.all(np.isfinite(controls))
            and math.isfinite(center)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _result_is_complete(
    result: dict,
    anchor: dict | None,
    sample_sha256: str,
) -> bool:
    try:
        N = int(result["N"])
        m = int(result["m"])
        center = _scalar(result["f_opt"])
        expected_grid = _width_grid(N)
        b_default = expected_grid[-1]
        points = {int(point["b"]): point for point in result["points"]}
    except (KeyError, TypeError, ValueError):
        return False
    if anchor is None or not _anchor_is_valid(anchor, N, sample_sha256):
        return False
    if (
        int(result.get("b_default", -1)) != b_default
        or result.get("b_grid") != expected_grid
        or result.get("fixed_sample_sha256") != sample_sha256
        or not math.isclose(center, _scalar(anchor["f_opt"]),
                            rel_tol=0.0, abs_tol=1e-12)
        or m != 5 * N
        or len(points) != len(result.get("points", ()))
        or set(points) != set(expected_grid)
    ):
        return False
    for point in points.values():
        deltas = np.asarray(point.get("deltas", ()), dtype=float)
        if deltas.shape != (m,) or not np.all(np.isfinite(deltas)):
            return False
        interval = subsampling_interval_from_deltas(deltas, center, N, LEVEL)
        if (
            int(point.get("rank_lo", -1)) != int(interval["rank_lo"])
            or int(point.get("rank_hi", -1)) != int(interval["rank_hi"])
        ):
            return False
        for key in ("lo", "hi", "width"):
            if not math.isclose(
                float(point.get(key, math.nan)), float(interval[key]),
                rel_tol=1e-12, abs_tol=1e-12,
            ):
                return False
    return True


def _replication_entry(data: dict, replication: int) -> dict | None:
    return next(
        (item for item in data["replications"]
         if int(item["replication"]) == int(replication)),
        None,
    )


def _completed_blocks(data: dict) -> set[tuple[int, int]]:
    completed = set()
    sample_sha256 = data.get("fixed_sample", {}).get("sha256", "")
    anchors = data.get("anchors", {})
    for replication in data.get("replications", ()):
        outer_id = int(replication["replication"])
        for result in replication.get("results", ()):
            N = int(result["N"])
            if _result_is_complete(result, anchors.get(str(N)), sample_sha256):
                completed.add((outer_id, N))
    return completed


def _validate_resume(
    data: dict,
    replications: int,
    sample_sizes: tuple[int, ...],
    root_seed: int,
    samples: np.ndarray,
) -> None:
    expected = {
        "version": 3,
        "R": int(replications),
        "sample_sizes": list(sample_sizes),
        "root_seed": int(root_seed),
        "q": int(INFERENCE_NINTERVALS),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError("resume file has incompatible {}".format(key))
    if not math.isclose(float(data.get("level", math.nan)), LEVEL,
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("resume file has an incompatible nominal level")
    if data.get("seed_plan") != _seed_plan(replications, root_seed):
        raise ValueError("resume file has an incompatible seed plan")
    expected_fixed_sample = {
        "scenario_seed": int(INFERENCE_SEED),
        "size": int(samples.shape[0]),
        "shape": list(samples.shape),
        "sha256": _sample_sha256(samples),
    }
    if data.get("fixed_sample") != expected_fixed_sample:
        raise ValueError("resume file has an incompatible fixed scenario sample")
    expected_grids = {str(N): _width_grid(N) for N in sample_sizes}
    if data.get("grid", {}).get("values_by_N") != expected_grids:
        raise ValueError("resume file has an incompatible subsample-size grid")
    meta = data.get("meta", {})
    expected_meta = {
        "model": "FedBatchReactor",
        "sampler": "UniformRelativeSampler",
        "m_expr": "5N",
        "resolver": "ipopt-warmstart from the matching full-sample control",
        "solver_tolerance": float(TOL_INFERENCE),
        "design": "conditional_on_one_fixed_full_sample",
        "full_sample_reused_across_repetitions": True,
        "subsampling_repetitions_independent": True,
        "outer_replications_independent": False,
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise ValueError("resume file has incompatible metadata field {}"
                             .format(key))

    anchors = data.get("anchors")
    if not isinstance(anchors, dict):
        raise ValueError("resume file has no anchor mapping")
    unknown_anchors = sorted(set(anchors) - {str(N) for N in sample_sizes})
    if unknown_anchors:
        raise ValueError("resume file contains anchors for unexpected sample sizes")
    for key, anchor in anchors.items():
        if not _anchor_is_valid(
            anchor, int(key), expected_fixed_sample["sha256"]
        ):
            raise ValueError("resume file contains an invalid N={} anchor".format(key))

    outer_ids = [int(item["replication"]) for item in data["replications"]]
    if len(outer_ids) != len(set(outer_ids)):
        raise ValueError("resume file has duplicate replication identifiers")
    if any(outer_id < 0 or outer_id >= replications for outer_id in outer_ids):
        raise ValueError("resume file has an out-of-range replication identifier")
    plan_by_replication = {
        int(item["replication"]): item for item in data["seed_plan"]
    }
    for replication in data["replications"]:
        outer_id = int(replication["replication"])
        planned = plan_by_replication[outer_id]
        if replication.get("subsample_seed_by_N") != planned["subsample_seed_by_N"]:
            raise ValueError(
                "resume file has incompatible seeds for replication {}"
                .format(outer_id)
            )
        results = replication.get("results", ())
        result_sizes = [int(result["N"]) for result in results]
        if len(result_sizes) != len(set(result_sizes)):
            raise ValueError("resume file contains duplicate results in a replication")
        if not set(result_sizes).issubset(set(sample_sizes)):
            raise ValueError("resume file contains an unexpected result sample size")
        for result in results:
            N = int(result["N"])
            if not _result_is_complete(
                result, anchors.get(str(N)), expected_fixed_sample["sha256"]
            ):
                raise ValueError(
                    "resume file contains an invalid replication {}, N={} block"
                    .format(outer_id, N)
                )


def _solve_missing_anchors(
    data: dict,
    sample_sizes: tuple[int, ...],
    samples: np.ndarray,
    json_path: Path,
) -> None:
    """Solve each missing full-sample anchor once and checkpoint immediately."""
    model = FedBatchReactor()
    model.nintervals = INFERENCE_NINTERVALS
    lower_bound = model.control_bounds[0][0]
    upper_bound = model.control_bounds[1][0]
    options = dict(ipopt_options(TOL_INFERENCE))
    options.update({"print_level": 0, "sb": "yes"})

    sample_digest = _sample_sha256(samples)
    for N in sample_sizes:
        existing = data["anchors"].get(str(N))
        if existing is not None:
            if not _anchor_is_valid(existing, N, sample_digest):
                raise ValueError("checkpoint contains an invalid N={} anchor".format(N))
            continue

        print("[width-mc] solving one fixed full-sample anchor for N={}".format(N),
              flush=True)
        with build_lock:
            saa = ensemblecontrol.SAAProblem(
                model,
                samples[:N],
                MultipleShooting=False,
                ipopt_options=options,
            )
        w_opt, f_opt = saa.solve()
        controls = project_interior(
            saa.control_matrix(w_opt), lower_bound, upper_bound)
        anchor = {
            "N": int(N),
            "f_opt": _scalar(f_opt),
            "projected_controls": np.asarray(controls, dtype=float).tolist(),
            "scenario_seed": int(INFERENCE_SEED),
            "fixed_sample_sha256": sample_digest,
        }
        if not _anchor_is_valid(anchor, N, sample_digest):
            raise RuntimeError("full-sample solve produced an invalid anchor")
        data["anchors"][str(N)] = anchor
        save_width_run(data, json_path)


def _silence_worker_output() -> None:
    """Discard verbose CasADi timing tables in spawned worker processes."""
    sys.stdout.flush()
    sys.stderr.flush()
    descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())
    os.close(descriptor)


def _initialize_worker(
    max_sample_size: int,
    expected_sample_sha256: str,
    anchors: dict[str, dict],
    silence_output: bool,
) -> None:
    """Rebuild the fixed full SAA in a process without optimizing its anchor."""
    global _WORKER_SAA, _WORKER_ANCHORS, _WORKER_SAMPLE_SHA256
    if silence_output:
        _silence_worker_output()

    samples = _fixed_samples(max_sample_size)
    sample_digest = _sample_sha256(samples)
    if sample_digest != expected_sample_sha256:
        raise RuntimeError("worker regenerated a different fixed scenario sample")

    model = FedBatchReactor()
    model.nintervals = INFERENCE_NINTERVALS
    options = dict(ipopt_options(TOL_INFERENCE))
    options.update({"print_level": 0, "sb": "yes"})
    with build_lock:
        _WORKER_SAA = ensemblecontrol.SAAProblem(
            model,
            samples,
            MultipleShooting=False,
            ipopt_options=options,
            parallelization="serial",
            n_threads=1,
        )
    _WORKER_ANCHORS = anchors
    _WORKER_SAMPLE_SHA256 = sample_digest


def _run_block(task: dict) -> dict:
    """Repeat subset draws/solves once; never solve a full-sample anchor."""
    if _WORKER_SAA is None or _WORKER_SAMPLE_SHA256 is None:
        raise RuntimeError("worker has not been initialized")
    N = int(task["N"])
    try:
        anchor = _WORKER_ANCHORS[str(N)]
    except KeyError as error:
        raise RuntimeError("worker is missing the N={} anchor".format(N)) from error
    if not _anchor_is_valid(anchor, N, _WORKER_SAMPLE_SHA256):
        raise RuntimeError("worker received an invalid N={} anchor".format(N))
    center = _scalar(anchor["f_opt"])
    controls = np.asarray(anchor["projected_controls"], dtype=float)

    def resolve(indices):
        with build_lock:
            subproblem = _WORKER_SAA.subproblem(indices)
            subproblem.initial_decisions = subproblem.initial_from_controls(controls)
        return subproblem.solve()[1]

    grid = _width_grid(N)
    b_default = grid[-1]
    m = 5 * N
    ordered = nested_index_sets(
        N,
        b_default,
        m,
        np.random.default_rng(int(task["subsample_seed"])),
    )
    cache: dict[tuple[int, ...], float] = {}
    points = []
    for b in grid:
        deltas = deltas_for_block_size(
            ordered, b, center, resolve, cache=cache)
        interval = subsampling_interval_from_deltas(deltas, center, N, LEVEL)
        points.append({
            "b": int(b),
            "deltas": deltas.tolist(),
            "rank_lo": int(interval["rank_lo"]),
            "rank_hi": int(interval["rank_hi"]),
            "lo": float(interval["lo"]),
            "hi": float(interval["hi"]),
            "width": float(interval["width"]),
        })

    result = {
        "N": N,
        "f_opt": center,
        "fixed_sample_sha256": _WORKER_SAMPLE_SHA256,
        "m": m,
        "b_default": b_default,
        "b_grid": grid,
        "points": points,
    }
    if not _result_is_complete(result, anchor, _WORKER_SAMPLE_SHA256):
        raise RuntimeError("worker produced an invalid completed block")
    return {
        "replication": int(task["replication"]),
        "subsample_seed": int(task["subsample_seed"]),
        "fixed_sample_sha256": _WORKER_SAMPLE_SHA256,
        "result": result,
    }


def _store_block(data: dict, block: dict) -> None:
    outer_id = int(block["replication"])
    planned = data["seed_plan"][outer_id]
    N = int(block["result"]["N"])
    sample_digest = data["fixed_sample"]["sha256"]
    anchor = data["anchors"].get(str(N))
    if block.get("fixed_sample_sha256") != sample_digest:
        raise ValueError("worker returned an unexpected fixed sample identity")
    if int(block["subsample_seed"]) != int(planned["subsample_seed_by_N"][str(N)]):
        raise ValueError("worker returned an unexpected subsampling seed")
    if not _result_is_complete(block["result"], anchor, sample_digest):
        raise ValueError("worker returned an invalid completed block")

    replication = _replication_entry(data, outer_id)
    if replication is None:
        replication = {
            "replication": outer_id,
            "subsample_seed_by_N": dict(planned["subsample_seed_by_N"]),
            "results": [],
        }
        data["replications"].append(replication)
        data["replications"].sort(key=lambda item: int(item["replication"]))
    prior = next(
        (result for result in replication["results"] if int(result["N"]) == N),
        None,
    )
    if prior is not None:
        if _result_is_complete(prior, anchor, sample_digest):
            raise ValueError("refusing to overwrite a completed block")
        replication["results"].remove(prior)
    replication["results"].append(block["result"])
    replication["results"].sort(key=lambda item: int(item["N"]))


def _all_complete(data: dict) -> bool:
    expected = {
        (replication, N)
        for replication in range(int(data["R"]))
        for N in data["sample_sizes"]
    }
    return _completed_blocks(data) == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--R", type=int, default=DEFAULT_REPLICATIONS,
                        help="independent subsampling repetitions (default: 30)")
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int, choices=DEFAULT_SAMPLE_SIZES,
        default=list(DEFAULT_SAMPLE_SIZES),
        help="sample sizes to include (default: 32 64)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=("outer process workers (default: {}; every optimization remains "
              "serial)".format(DEFAULT_WORKERS)),
    )
    parser.add_argument("--root-seed", type=int,
                        default=SUBSAMPLING_WIDTH_ROOT_SEED)
    parser.add_argument(
        "--resume", type=Path,
        help=("continue an existing subsampling_width_monte_carlo.json; the "
              "default results/revision checkpoint is resumed automatically"),
    )
    parser.add_argument(
        "--outdir", type=Path,
        help="fresh-run output directory; default: results/revision",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=FORMATS, default=list(FORMATS),
        help="figure formats written after completion (default: png pdf)",
    )
    args = parser.parse_args()

    if args.R < 1:
        parser.error("--R must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.resume is not None and args.outdir is not None:
        parser.error("--resume and --outdir cannot be used together")
    sizes = tuple(int(N) for N in args.sample_sizes)
    if len(sizes) != len(set(sizes)):
        parser.error("sample sizes must be distinct")
    sizes = tuple(N for N in DEFAULT_SAMPLE_SIZES if N in sizes)
    if not sizes:
        parser.error("at least one sample size is required")
    samples = _fixed_samples(max(sizes))
    started = time.monotonic()

    default_run_dir = Path(repo_root(__file__)) / "results" / "revision"
    resume_path = args.resume.resolve() if args.resume is not None else None
    if (
        resume_path is None
        and args.outdir is None
        and (default_run_dir / RUN_JSON).exists()
    ):
        resume_path = default_run_dir / RUN_JSON

    if resume_path is not None:
        json_path = resume_path
        data = load_width_run(json_path)
        try:
            _validate_resume(data, args.R, sizes, args.root_seed, samples)
        except ValueError as error:
            parser.error(str(error))
        run_dir = json_path.parent
        data["complete"] = False
        save_width_run(data, json_path)
    else:
        run_dir = default_run_dir if args.outdir is None else args.outdir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        json_path = run_dir / RUN_JSON
        if json_path.exists():
            parser.error("output checkpoint already exists; continue it with --resume")
        data = _new_run(args.R, sizes, args.root_seed, samples)
        save_width_run(data, json_path)

    _solve_missing_anchors(data, sizes, samples, json_path)
    completed = _completed_blocks(data)
    tasks = []
    for planned in data["seed_plan"]:
        outer_id = int(planned["replication"])
        for N in sizes:
            if (outer_id, N) in completed:
                continue
            tasks.append({
                "replication": outer_id,
                "N": N,
                "subsample_seed": int(planned["subsample_seed_by_N"][str(N)]),
            })

    total_blocks = args.R * len(sizes)
    total_subsample_solves = args.R * sum(
        len(_width_grid(N)) * 5 * N for N in sizes)
    print(
        "[width-mc] sizes={} R={} workers={} ({} nominal subsample solves; "
        "{} fixed anchors reused)"
        .format(list(sizes), args.R, min(args.workers, max(1, len(tasks))),
                total_subsample_solves, len(sizes)),
        flush=True,
    )
    if completed:
        print("[width-mc] resuming after {}/{} completed blocks"
              .format(len(completed), total_blocks), flush=True)

    if tasks:
        workers = min(args.workers, len(tasks))
        initializer_args = (
            max(sizes), data["fixed_sample"]["sha256"], data["anchors"]
        )
        if workers == 1:
            _initialize_worker(*initializer_args, silence_output=False)
            for task in tasks:
                block = _run_block(task)
                _store_block(data, block)
                save_width_run(data, json_path)
                completed.add((int(block["replication"]), int(block["result"]["N"])))
                print(
                    "[width-mc] completed block {}/{}: replication {:02d}, N={} "
                    "({:.1f} min)".format(
                        len(completed), total_blocks, block["replication"],
                        block["result"]["N"], (time.monotonic() - started) / 60.0),
                    flush=True,
                )
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(*initializer_args, True),
            ) as pool:
                futures = {pool.submit(_run_block, task): task for task in tasks}
                try:
                    for future in as_completed(futures):
                        block = future.result()
                        _store_block(data, block)
                        save_width_run(data, json_path)
                        completed.add((int(block["replication"]),
                                       int(block["result"]["N"])))
                        print(
                            "[width-mc] completed block {}/{}: replication {:02d}, "
                            "N={} ({:.1f} min)".format(
                                len(completed), total_blocks, block["replication"],
                                block["result"]["N"],
                                (time.monotonic() - started) / 60.0),
                            flush=True,
                        )
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise

    data["complete"] = _all_complete(data)
    save_width_run(data, json_path)
    if not data["complete"]:
        raise RuntimeError("run ended without every requested block")

    summary_path = save_width_summary(data, run_dir / SUMMARY_CSV)
    figure_paths = plot_mean_subsampling_width(
        data, run_dir, formats=tuple(args.formats))
    print("[width-mc] raw data: {}".format(json_path), flush=True)
    print("[width-mc] summary: {}".format(summary_path), flush=True)
    for path in figure_paths:
        print("[width-mc] figure: {}".format(path), flush=True)


if __name__ == "__main__":
    main()
