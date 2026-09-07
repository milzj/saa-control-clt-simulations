"""Conditional subsampling-width diagnostics for the paper revision.

The scenario sample and its full-sample SAA optimum are held fixed. For each
subsample size ``b_N``, only the random subset-index stage is repeated, and the
resulting nominal confidence-interval widths are averaged. This module holds
the complete reusable pipeline: subsample construction, exact empirical
quantiles, JSON checkpointing, tabulation, and the approved paper figures.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable, MutableMapping, Sequence

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ensemblecontrol.inference_plotting import configure_style


ALGORITHM = "subsampling_width_monte_carlo"
DEFAULT_GRID_SIZE = 7

__all__ = [
    "ALGORITHM",
    "DEFAULT_GRID_SIZE",
    "deltas_for_block_size",
    "empirical_quantile_rank",
    "load_width_run",
    "lower_empirical_quantile",
    "nested_index_sets",
    "plot_mean_subsampling_width",
    "save_width_run",
    "save_width_summary",
    "subsample_size_grid",
    "subsampling_cis_from_deltas",
    "subsampling_interval_from_deltas",
    "summarize_widths",
]


def subsample_size_grid(N: int, point_count: int = DEFAULT_GRID_SIZE) -> list[int]:
    """Return an equally spaced admissible grid ending at ``floor(N**(6/7))``.

    Let ``B_N = floor(N**(6/7))`` and
    ``d_N = max(1, floor(B_N / point_count))``. The candidates are

    ``B_N - (point_count-j) d_N``, ``j=1,...,point_count``,

    after nonpositive values and duplicates are removed. Each finite-sample
    grid is arithmetic and ends at the manuscript choice ``B_N``.
    """
    N = int(N)
    point_count = int(point_count)
    if N < 2:
        raise ValueError("N must be at least two")
    if point_count < 2:
        raise ValueError("point_count must be at least two")

    upper = int(math.floor(N ** (6.0 / 7.0) + 1e-9))
    upper = min(N - 1, max(1, upper))
    step = max(1, upper // point_count)
    proposed = [
        upper - (point_count - j) * step
        for j in range(1, point_count + 1)
    ]
    return sorted({b for b in proposed if 1 <= b < N})


def _as_probability_fraction(probability: float | str | Fraction) -> Fraction:
    """Represent a user-facing decimal probability exactly."""
    try:
        value = (
            probability
            if isinstance(probability, Fraction)
            else Fraction(str(probability))
        )
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("probability must be numeric") from error
    if not Fraction(0) <= value <= Fraction(1):
        raise ValueError("probability must lie in [0, 1]")
    return value


def empirical_quantile_rank(
    m: int, probability: float | str | Fraction
) -> int:
    """Return the exact one-based lower-quantile rank ``ceil(m p)``."""
    m = int(m)
    if m < 1:
        raise ValueError("m must be positive")
    if isinstance(probability, (float, np.floating)):
        probability_value = float(probability)
        if (
            not math.isfinite(probability_value)
            or not 0.0 <= probability_value <= 1.0
        ):
            raise ValueError("probability must lie in [0, 1]")
        raw_rank = m * probability_value
        nearest = round(raw_rank)
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(raw_rank))
        if abs(raw_rank - nearest) <= tolerance:
            raw_rank = float(nearest)
        rank = int(math.ceil(raw_rank))
        return min(max(rank, 1), m)

    probability_fraction = _as_probability_fraction(probability)
    numerator = m * probability_fraction.numerator
    denominator = probability_fraction.denominator
    rank = (numerator + denominator - 1) // denominator
    return min(max(rank, 1), m)


def lower_empirical_quantile(
    values: Sequence[float], probability: float | str | Fraction
) -> float:
    """Return the order statistic with one-based rank ``ceil(m p)``."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.ndim != 1 or ordered.size == 0:
        raise ValueError("values must be a nonempty one-dimensional sequence")
    return float(ordered[empirical_quantile_rank(ordered.size, probability) - 1])


def subsampling_interval_from_deltas(
    deltas: Sequence[float],
    f_opt: float,
    N: int,
    level: float | str | Fraction = 0.95,
) -> dict[str, float | int]:
    """Compute the root-inverted subsampling interval from stored statistics."""
    N = int(N)
    level_fraction = _as_probability_fraction(level)
    level_value = float(level_fraction)
    if N < 1:
        raise ValueError("N must be positive")
    if not Fraction(0) < level_fraction < Fraction(1):
        raise ValueError("level must lie strictly between zero and one")

    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("deltas must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("deltas must contain only finite values")

    beta = 1 - level_fraction
    p_lo = beta / 2
    p_hi = 1 - beta / 2
    q_lo = lower_empirical_quantile(values, p_lo)
    q_hi = lower_empirical_quantile(values, p_hi)
    center = float(np.asarray(f_opt).squeeze())
    if not math.isfinite(center):
        raise ValueError("f_opt must be finite")
    lo = center - q_hi / math.sqrt(N)
    hi = center - q_lo / math.sqrt(N)
    return {
        "N": N,
        "m": int(values.size),
        "level": level_value,
        "rank_lo": empirical_quantile_rank(values.size, p_lo),
        "rank_hi": empirical_quantile_rank(values.size, p_hi),
        "quantile_lo": q_lo,
        "quantile_hi": q_hi,
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
    }


def subsampling_cis_from_deltas(
    deltas: Sequence[float],
    f_opt: float,
    N: int,
    levels: Sequence[float | str | Fraction] = (0.90, 0.95, 0.99),
) -> dict:
    """Return exact-rank subsampling CIs in EnsembleControl's plot schema."""
    level_fractions = tuple(_as_probability_fraction(level) for level in levels)
    if not level_fractions:
        raise ValueError("levels must be nonempty")
    level_values = tuple(float(level) for level in level_fractions)

    intervals = [
        subsampling_interval_from_deltas(deltas, f_opt, N, level)
        for level in level_fractions
    ]
    center = float(np.asarray(f_opt).squeeze())
    return {
        "N": int(N),
        "Jhat": center,
        "m": int(np.asarray(deltas).size),
        "levels": {
            level: {
                "quantile_lo": interval["quantile_lo"],
                "quantile_hi": interval["quantile_hi"],
                "rank_lo": interval["rank_lo"],
                "rank_hi": interval["rank_hi"],
                "lo": interval["lo"],
                "hi": interval["hi"],
            }
            for level, interval in zip(level_values, intervals)
        },
    }


def nested_index_sets(
    N: int, b_max: int, m: int, rng: np.random.Generator
) -> list[list[int]]:
    """Draw size-``b_max`` subsets whose prefixes couple smaller ``b`` values."""
    N, b_max, m = int(N), int(b_max), int(m)
    if not 0 < b_max < N:
        raise ValueError("b_max must satisfy 0 < b_max < N")
    if m < 1:
        raise ValueError("m must be positive")
    return [
        rng.choice(N, size=b_max, replace=False).astype(int).tolist()
        for _ in range(m)
    ]


def deltas_for_block_size(
    ordered_index_sets: Sequence[Sequence[int]],
    b: int,
    f_opt: float,
    resolve: Callable[[Sequence[int]], float],
    *,
    cache: MutableMapping[tuple[int, ...], float] | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> np.ndarray:
    """Resolve nested prefixes and return ``sqrt(b) (J_I^* - J_N^*)``.

    Duplicate subsets are solved once but retain their original multiplicity.
    ``progress(done, total, unique_solves)`` is called after every replicate.
    """
    b = int(b)
    if b < 1:
        raise ValueError("b must be positive")
    if not ordered_index_sets:
        raise ValueError("ordered_index_sets must be nonempty")
    if any(len(indices) < b for indices in ordered_index_sets):
        raise ValueError("every ordered index set must contain at least b entries")

    memo = {} if cache is None else cache
    deltas = np.empty(len(ordered_index_sets), dtype=float)
    center = float(np.asarray(f_opt).squeeze())
    if not math.isfinite(center):
        raise ValueError("f_opt must be finite")
    unique_solves = 0
    for repetition, ordered in enumerate(ordered_index_sets):
        indices = [int(index) for index in ordered[:b]]
        key = tuple(sorted(indices))
        if key not in memo:
            value = float(np.asarray(resolve(indices)).squeeze())
            if not math.isfinite(value):
                raise ValueError("subsample solve returned a nonfinite optimal value")
            memo[key] = value
            unique_solves += 1
        deltas[repetition] = math.sqrt(b) * (memo[key] - center)
        if progress is not None:
            progress(repetition + 1, len(ordered_index_sets), unique_solves)
    return deltas


def save_width_run(data: dict, path: str | os.PathLike[str]) -> Path:
    """Atomically checkpoint a repeated-subsampling width study as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=1, allow_nan=False)
    os.replace(temporary, target)
    return target


def load_width_run(path: str | os.PathLike[str]) -> dict:
    """Load and minimally validate a repeated-subsampling width study."""
    with Path(path).open() as stream:
        data = json.load(stream)
    if data.get("algorithm") != ALGORITHM:
        raise ValueError("not a subsampling-width Monte Carlo run")
    if "replications" not in data:
        raise ValueError("width run has no replications")
    return data


def _as_run(run_or_path: dict | str | os.PathLike[str]) -> dict:
    return (
        load_width_run(run_or_path)
        if isinstance(run_or_path, (str, os.PathLike))
        else run_or_path
    )


def summarize_widths(
    run_or_path: dict | str | os.PathLike[str],
) -> list[dict]:
    """Return mean, standard deviation, and standard error for each ``(N,b)``."""
    run = _as_run(run_or_path)
    conditional = bool(
        (run.get("meta") or {}).get("full_sample_reused_across_repetitions")
    )
    groups: dict[tuple[int, int], dict] = {}
    replication_ids: set[int] = set()
    center_by_N: dict[int, float] = {}
    for replication in run.get("replications", ()):
        replication_id = int(replication["replication"])
        if replication_id in replication_ids:
            raise ValueError("replication identifiers must be unique")
        replication_ids.add(replication_id)
        seen_in_replication: set[tuple[int, int]] = set()
        for result in replication.get("results", ()):
            N = int(result["N"])
            if conditional:
                center = float(result["f_opt"])
                if not math.isfinite(center):
                    raise ValueError("fixed full-sample centers must be finite")
                prior_center = center_by_N.setdefault(N, center)
                if not math.isclose(center, prior_center, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        "conditional repetitions must reuse one full-sample center"
                    )
            b_default = int(result["b_default"])
            m = int(result["m"])
            for point in result.get("points", ()):
                b = int(point["b"])
                key = (N, b)
                if key in seen_in_replication:
                    raise ValueError("a replication contains a duplicate (N, b) result")
                seen_in_replication.add(key)
                width = float(point["width"])
                if not math.isfinite(width) or width < 0.0:
                    raise ValueError("interval widths must be finite and nonnegative")
                group = groups.setdefault(
                    key,
                    {
                        "N": N,
                        "b": b,
                        "b_default": b_default,
                        "m": m,
                        "widths": [],
                    },
                )
                if group["b_default"] != b_default or group["m"] != m:
                    raise ValueError("inconsistent repeated-sample metadata")
                group["widths"].append(width)

    rows: list[dict] = []
    for group in sorted(groups.values(), key=lambda item: (item["N"], item["b"])):
        widths = np.asarray(group.pop("widths"), dtype=float)
        count = int(widths.size)
        standard_deviation = float(np.std(widths, ddof=1)) if count > 1 else 0.0
        rows.append(
            {
                "N": group["N"],
                "m": group["m"],
                "b": group["b"],
                "b_over_N": float(group["b"]) / float(group["N"]),
                "replications": count,
                "mean_width": float(np.mean(widths)),
                "sd_width": standard_deviation,
                "se_mean_width": standard_deviation / math.sqrt(count),
            }
        )
    return rows


def save_width_summary(
    run_or_path: dict | str | os.PathLike[str],
    path: str | os.PathLike[str],
) -> Path:
    """Write one mean-width summary row per ``(N,b_N)`` to CSV."""
    rows = summarize_widths(run_or_path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "N",
        "m",
        "b",
        "b_over_N",
        "replications",
        "mean_width",
        "sd_width",
        "se_mean_width",
    ]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _save_formats(
    figure, base: Path, formats: Iterable[str], dpi: int
) -> list[Path]:
    written = []
    for suffix in formats:
        path = base.with_suffix("." + str(suffix).lstrip("."))
        figure.savefig(path, dpi=dpi)
        written.append(path)
    return written


def plot_mean_subsampling_width(
    run_or_path: dict | str | os.PathLike[str],
    outdir: str | os.PathLike[str],
    *,
    prefix: str = "fed-batch-mean-subsampling-width",
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 180,
) -> list[Path]:
    """Write one approved paper-style mean-width figure per sample size."""
    configure_style()
    rows = summarize_widths(run_or_path)
    if not rows:
        raise ValueError("cannot plot a run with no completed replications")

    by_N: dict[int, list[dict]] = {}
    for row in rows:
        by_N.setdefault(int(row["N"]), []).append(row)
    replication_counts = {int(row["replications"]) for row in rows}
    if len(replication_counts) != 1:
        raise ValueError("every (N, b) needs the same number of replications")
    replications = replication_counts.pop()

    means = np.asarray([row["mean_width"] for row in rows], dtype=float)
    y_min = float(np.min(means))
    y_max = float(np.max(means))
    y_span = max(y_max - y_min, 1e-8)
    shared_ylim = (max(0.0, y_min - 0.08 * y_span), y_max + 0.08 * y_span)

    mean_handle = mlines.Line2D(
        [],
        [],
        color="C0",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="C0",
        markeredgewidth=1.6,
        linewidth=2.1,
        label=rf"mean over {replications} subsampling repetitions",
    )

    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    formats = tuple(formats)
    written: list[Path] = []
    for N in sorted(by_N):
        panel = sorted(by_N[N], key=lambda row: int(row["b"]))
        b_values = np.asarray([row["b"] for row in panel], dtype=int)
        panel_means = np.asarray([row["mean_width"] for row in panel], dtype=float)

        figure, axis = plt.subplots()
        axis.plot(
            b_values,
            panel_means,
            color="C0",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="C0",
            markeredgewidth=1.6,
            markersize=6.3,
            linewidth=2.1,
        )
        x_span = max(float(np.ptp(b_values)), 1.0)
        axis.set_xlim(
            float(b_values.min()) - 0.06 * x_span,
            float(b_values.max()) + 0.06 * x_span,
        )
        axis.set_ylim(*shared_ylim)
        axis.set_xticks(b_values)
        axis.set_xlabel(r"subsample size $b_N$")
        axis.set_ylabel(r"mean $95\%$ subsampling-CI width")
        axis.grid(True, which="both", alpha=0.3)

        metadata_handles = [
            mpatches.Patch(color="none", label=rf"$N = {N}$"),
            mpatches.Patch(color="none", label=rf"$m_N = {panel[0]['m']}$"),
        ]
        axis.legend(
            handles=[mean_handle, *metadata_handles],
            loc="lower left" if N == 32 else "upper right",
            frameon=True,
            framealpha=1.0,
        )
        figure.tight_layout()
        base = destination / f"{prefix}-N{N}-R{replications}"
        written.extend(_save_formats(figure, base, formats, dpi))
        plt.close(figure)
    return written
