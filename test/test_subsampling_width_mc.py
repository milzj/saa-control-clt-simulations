import json
import math
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pytest

from saa_clt.subsampling_width_mc import (
    ALGORITHM,
    deltas_for_block_size,
    empirical_quantile_rank,
    load_width_run,
    nested_index_sets,
    plot_mean_subsampling_width,
    save_width_run,
    save_width_summary,
    subsample_size_grid,
    subsampling_cis_from_deltas,
    subsampling_interval_from_deltas,
    summarize_widths,
)


def _synthetic_run():
    widths = {
        0: {7: 1.0, 19: 2.0},
        1: {7: 2.0, 19: 4.0},
        2: {7: 3.0, 19: 6.0},
    }
    return {
        "algorithm": ALGORITHM,
        "R": 3,
        "sample_sizes": [32],
        "level": 0.95,
        "meta": {"full_sample_reused_across_repetitions": True},
        "replications": [
            {
                "replication": replication,
                "results": [
                    {
                        "N": 32,
                        "f_opt": -30.0,
                        "m": 160,
                        "b_default": 19,
                        "points": [
                            {"b": b, "width": width}
                            for b, width in sorted(by_b.items())
                        ],
                    }
                ],
            }
            for replication, by_b in widths.items()
        ],
    }


def _synthetic_two_size_run():
    run = _synthetic_run()
    run["sample_sizes"] = [32, 64]
    for replication in run["replications"]:
        replication_id = int(replication["replication"])
        replication["results"].append(
            {
                "N": 64,
                "f_opt": -31.0,
                "m": 320,
                "b_default": 35,
                "points": [
                    {"b": 5, "width": 0.5 + 0.1 * replication_id},
                    {"b": 35, "width": 1.0 + 0.1 * replication_id},
                ],
            }
        )
    return run


def test_subsample_size_grids_are_arithmetic_and_end_at_paper_choice():
    expected = {
        8: [1, 2, 3, 4, 5],
        16: [4, 5, 6, 7, 8, 9, 10],
        32: [7, 9, 11, 13, 15, 17, 19],
        64: [5, 10, 15, 20, 25, 30, 35],
    }
    for N, grid in expected.items():
        assert subsample_size_grid(N) == grid
        assert grid[-1] == math.floor(N ** (6.0 / 7.0) + 1e-9)
        if len(grid) > 2:
            assert len(set(np.diff(grid))) == 1


def test_grid_sequences_diverge_and_are_small_relative_to_N():
    small = subsample_size_grid(10**7)
    large = subsample_size_grid(10**14)
    assert len(small) == len(large) == 7
    assert all(b_large > b_small for b_small, b_large in zip(small, large))
    assert large[-1] / 10**14 < small[-1] / 10**7


def test_empirical_rank_snaps_floating_point_near_integer():
    probability = (1.0 - 0.95) / 2.0
    assert [
        empirical_quantile_rank(m, probability) for m in (40, 80, 160, 320)
    ] == [1, 2, 4, 8]
    assert empirical_quantile_rank(10, 0.21) == 3


def test_empirical_rank_uses_exact_decimal_probabilities():
    assert [
        empirical_quantile_rank(m, Fraction(1, 40))
        for m in (40, 80, 160, 320)
    ] == [1, 2, 4, 8]
    assert [
        empirical_quantile_rank(m, Fraction(39, 40))
        for m in (40, 80, 160, 320)
    ] == [39, 78, 156, 312]


def test_interval_uses_manuscript_lower_empirical_quantiles():
    deltas = np.arange(1.0, 41.0)
    interval = subsampling_interval_from_deltas(
        deltas, f_opt=10.0, N=4, level=0.95
    )
    assert interval["rank_lo"] == 1
    assert interval["rank_hi"] == 39
    assert interval["quantile_lo"] == 1.0
    assert interval["quantile_hi"] == 39.0
    assert interval["lo"] == 10.0 - 39.0 / 2.0
    assert interval["hi"] == 10.0 - 1.0 / 2.0

    compatible = subsampling_cis_from_deltas(
        deltas, f_opt=10.0, N=4, levels=("0.95",)
    )
    assert compatible["levels"][0.95]["rank_lo"] == 1
    assert compatible["levels"][0.95]["rank_hi"] == 39

    rational_level = subsampling_cis_from_deltas(
        np.arange(1.0, 4.0), f_opt=10.0, N=4, levels=(Fraction(1, 3),)
    )
    assert rational_level["levels"][float(Fraction(1, 3))]["rank_lo"] == 1
    assert rational_level["levels"][float(Fraction(1, 3))]["rank_hi"] == 2


def test_nested_draws_and_duplicate_subset_cache():
    draws = nested_index_sets(5, b_max=3, m=20, rng=np.random.default_rng(7))
    assert len(draws) == 20
    assert all(len(row) == len(set(row)) == 3 for row in draws)
    assert all(set(row[:2]).issubset(row) for row in draws)

    calls = []

    def resolve(indices):
        calls.append(tuple(sorted(indices)))
        return sum(indices)

    cache = {}
    deltas = deltas_for_block_size(draws, 1, 0.0, resolve, cache=cache)
    assert deltas.shape == (20,)
    assert len(calls) == len(set(tuple(sorted(row[:1])) for row in draws))
    assert len(cache) == len(calls)


def test_summary_contains_mean_width_statistics_without_mc_interval():
    rows = summarize_widths(_synthetic_run())
    first = rows[0]
    assert (first["N"], first["b"], first["replications"]) == (32, 7, 3)
    assert first["mean_width"] == 2.0
    assert first["sd_width"] == 1.0
    assert first["se_mean_width"] == pytest.approx(1.0 / math.sqrt(3.0))
    assert not any(key.startswith("mc_") for key in first)
    assert "selected" not in first


def test_single_replication_has_zero_sd_and_se():
    run = _synthetic_run()
    run["replications"] = run["replications"][:1]
    row = summarize_widths(run)[0]
    assert row["sd_width"] == 0.0
    assert row["se_mean_width"] == 0.0


def test_duplicate_replication_identifiers_are_rejected():
    run = _synthetic_run()
    run["replications"][1]["replication"] = 0
    with pytest.raises(ValueError, match="identifiers"):
        summarize_widths(run)


def test_duplicate_result_within_replication_is_rejected():
    run = _synthetic_run()
    run["replications"][0]["results"][0]["points"].append(
        {"b": 7, "width": 1.5}
    )
    with pytest.raises(ValueError, match="duplicate"):
        summarize_widths(run)


def test_conditional_repetitions_must_share_full_sample_center():
    run = _synthetic_run()
    run["replications"][1]["results"][0]["f_opt"] = -29.9
    with pytest.raises(ValueError, match="reuse one full-sample center"):
        summarize_widths(run)


def test_json_csv_and_single_panel_roundtrip(tmp_path):
    path = save_width_run(_synthetic_run(), tmp_path / "run.json")
    assert load_width_run(path) == _synthetic_run()
    assert not (tmp_path / "run.json.tmp").exists()
    with path.open() as stream:
        assert json.load(stream)["algorithm"] == ALGORITHM

    csv_path = save_width_summary(path, tmp_path / "summary.csv")
    header = csv_path.read_text().splitlines()[0].split(",")
    assert header == [
        "N",
        "m",
        "b",
        "b_over_N",
        "replications",
        "mean_width",
        "sd_width",
        "se_mean_width",
    ]
    assert not any(field.startswith("mc_") for field in header)

    figure_paths = plot_mean_subsampling_width(
        path, tmp_path, formats=("png",), dpi=72
    )
    assert [item.name for item in figure_paths] == [
        "fed-batch-mean-subsampling-width-N32-R3.png"
    ]
    assert figure_paths[0].stat().st_size > 0


def test_paper_plots_have_requested_legends_and_shared_scale(tmp_path, monkeypatch):
    figures = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: figures.append(figure))

    figure_paths = plot_mean_subsampling_width(
        _synthetic_two_size_run(), tmp_path, formats=("png",), dpi=72
    )

    assert [path.name for path in figure_paths] == [
        "fed-batch-mean-subsampling-width-N32-R3.png",
        "fed-batch-mean-subsampling-width-N64-R3.png",
    ]
    assert all(path.stat().st_size > 0 for path in figure_paths)
    assert len(figures) == 2
    axes = [figure.axes[0] for figure in figures]
    assert axes[0].get_ylim() == pytest.approx(axes[1].get_ylim())
    assert all(axis.get_title() == "" for axis in axes)
    assert all(
        axis.get_ylabel() == r"mean $95\%$ subsampling-CI width" for axis in axes
    )

    legend_text = [
        [text.get_text() for text in axis.get_legend().get_texts()]
        for axis in axes
    ]
    assert legend_text[0] == [
        "mean over 3 subsampling repetitions",
        r"$N = 32$",
        r"$m_N = 160$",
    ]
    assert legend_text[1] == [
        "mean over 3 subsampling repetitions",
        r"$N = 64$",
        r"$m_N = 320$",
    ]
    assert all(not axis.collections for axis in axes)
    assert all(len(axis.lines) == 1 for axis in axes)
    assert all(axis.lines[0].get_color() == "C0" for axis in axes)
    assert axes[0].get_legend()._loc == 3
    assert axes[1].get_legend()._loc == 1

    for figure in figures:
        real_close(figure)


def test_negative_or_nonfinite_width_is_rejected():
    for bad in (-1.0, np.nan):
        run = _synthetic_run()
        run["replications"][0]["results"][0]["points"][0]["width"] = bad
        with pytest.raises(ValueError, match="nonnegative"):
            summarize_widths(run)


def test_nonfinite_statistics_centers_and_solve_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        subsampling_interval_from_deltas(
            [0.0, np.nan], f_opt=1.0, N=2, level=0.95
        )
    with pytest.raises(ValueError, match="f_opt"):
        deltas_for_block_size([[0]], 1, np.inf, lambda _indices: 0.0)
    with pytest.raises(ValueError, match="nonfinite"):
        deltas_for_block_size([[0]], 1, 0.0, lambda _indices: np.inf)
