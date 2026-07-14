"""Unit tests for ``saa_clt.coverage_plotting`` and the two ``outputs`` helpers it
leans on (``load_coverage_intervals``, ``latest_run_dir``).

Everything runs against a small synthetic intervals JSON built the same way as
``test_coverage_intervals`` -- written by the real writer, so the tests pin the
writer/reader round-trip rather than a hand-rolled fixture -- and never touches the
multi-MB files under ``results/``.  No solving; ``conftest`` already forces Agg.
"""

import os

import numpy as np
import pytest
from scipy.stats import norm

from saa_clt.coverage_plotting import (plot_coverage_intervals, plot_interval_grid,
                                       INTERVALS_JSON)
from saa_clt.outputs import (save_coverage_intervals, load_coverage_intervals,
                             latest_run_dir)

LEVELS = (0.90, 0.95, 0.99)
SIZES = (4, 8)
R = 5
F_REF = -2.0


def _write_intervals(dirpath):
    """A synthetic coverage study (known centre/se per replication, symmetric plug-in
    endpoints per level), persisted with the real writer.  Returns the JSON path."""
    rng = np.random.default_rng(0)
    z = {lvl: float(norm.ppf(1.0 - (1.0 - lvl) / 2.0)) for lvl in LEVELS}
    bounds_by_N = {}
    for N in SIZES:
        jhat = -2.0 + rng.standard_normal(R)      # some straddle F_REF, some miss
        se = 0.1 + rng.random(R)                  # strictly positive
        bounds_by_N[N] = {lvl: {"lo": jhat - z[lvl] * se, "hi": jhat + z[lvl] * se}
                          for lvl in LEVELS}
    study = {"levels": list(LEVELS), "n_ref": 128, "f_ref": F_REF, "R": R,
             "sample_sizes": list(SIZES), "bounds_by_N": bounds_by_N}
    path = os.path.join(str(dirpath), INTERVALS_JSON)
    save_coverage_intervals(study, path, meta={"model": "X", "ci": "plugin"})
    return path


def test_load_coverage_intervals_round_trips(tmp_path):
    data = load_coverage_intervals(_write_intervals(tmp_path))

    assert data["levels"] == list(LEVELS)
    assert data["f_ref"] == F_REF
    assert data["R"] == R
    assert data["n_ref"] == 128
    assert [b["N"] for b in data["results"]] == sorted(SIZES)

    for blk in data["results"]:
        assert blk["lo"].shape == (len(LEVELS), R)
        assert blk["hi"].shape == (len(LEVELS), R)
        assert blk["J_hat_N"].shape == (R,)
        assert blk["se"].shape == (R,)
        # the loader hands back arrays, not the raw JSON lists
        assert isinstance(blk["lo"], np.ndarray)
        # symmetric plug-in CI: the centre sits midway between the endpoints
        np.testing.assert_allclose(0.5 * (blk["lo"][0] + blk["hi"][0]), blk["J_hat_N"])


def test_covers_derivable_from_endpoints(tmp_path):
    """The cover indicator is not stored; it must be recoverable from lo/hi/f_ref."""
    data = load_coverage_intervals(_write_intervals(tmp_path))
    for blk in data["results"]:
        covers = (blk["lo"] <= F_REF) & (F_REF <= blk["hi"])
        for j in range(len(LEVELS)):
            manual = [blk["lo"][j][i] <= F_REF <= blk["hi"][j][i] for i in range(R)]
            np.testing.assert_array_equal(covers[j], manual)
        # a wider (higher-level) interval covers whenever a narrower one does
        assert covers[2].sum() >= covers[1].sum() >= covers[0].sum()


def test_plot_writes_figures_and_csv(tmp_path):
    path = _write_intervals(tmp_path)
    written = plot_coverage_intervals(path, outdir=str(tmp_path), stamp="STAMP",
                                      level=0.95, n_show=R, formats=("png", "pdf"))

    base = os.path.join(str(tmp_path), "coverage_plugin_STAMP_intervals95")
    assert written == [base + ".png", base + ".pdf", base + ".csv"]
    for p in written:
        assert os.path.getsize(p) > 0

    lines = open(base + ".csv").read().splitlines()
    assert lines[0] == "N,replication,J_hat_N,se,lo,hi,covers"
    assert len(lines) == len(SIZES) * R + 1        # one row per (N, replication)

    rows = np.loadtxt(base + ".csv", delimiter=",", skiprows=1)
    assert set(rows[:, 0]) == set(SIZES)
    assert set(rows[:, 1]) == set(range(1, R + 1))
    assert set(np.unique(rows[:, 6])) <= {0.0, 1.0}
    # csv covers agrees with the endpoints it was derived from
    np.testing.assert_array_equal(rows[:, 6].astype(bool),
                                  (rows[:, 4] <= F_REF) & (F_REF <= rows[:, 5]))


def test_plot_accepts_a_run_dir(tmp_path):
    _write_intervals(tmp_path)
    written = plot_coverage_intervals(str(tmp_path), outdir=str(tmp_path),
                                      stamp="S", n_show=2, formats=("png",))
    assert written[0].endswith("coverage_plugin_S_intervals95.png")


@pytest.mark.parametrize("n_show, expected", [(10 ** 6, R), (2, 2), (R, R)])
def test_n_show_is_clamped_to_R(tmp_path, n_show, expected):
    path = _write_intervals(tmp_path)
    fig, axes = plot_coverage_intervals(path, n_show=n_show)
    try:
        # x runs 1..n over the plotted replications, so the axis pins the clamp
        assert np.asarray(axes).ravel()[0].get_xlim() == (0.5, expected + 0.5)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_unknown_level_raises(tmp_path):
    path = _write_intervals(tmp_path)
    with pytest.raises(ValueError, match="0.5"):
        plot_coverage_intervals(path, level=0.5)


def test_interval_grid_writes_one_figure_for_every_size(tmp_path):
    """All sizes share a single figure -- no per-N suffix, one file per format."""
    path = _write_intervals(tmp_path)
    written = plot_interval_grid(path, outdir=str(tmp_path), stamp="STAMP",
                                 level=0.95, n_show=R, formats=("png", "pdf"))
    base = os.path.join(str(tmp_path), "coverage_plugin_STAMP_intervals95_grid")
    assert written == [base + ".png", base + ".pdf"]
    for p in written:
        assert os.path.getsize(p) > 0


def test_interval_grid_default_gives_one_panel_per_replication(tmp_path):
    """group_by='replication': R panels total, every size drawn inside each."""
    path = _write_intervals(tmp_path)
    fig, axes = plot_interval_grid(path, n_show=R)
    try:
        assert len(axes) == R
        assert [ax.get_title() for ax in axes] == [str(i + 1) for i in range(R)]
        # one x position per size ...
        assert list(axes[0].get_xticks()) == list(range(len(SIZES)))
        # ... tick-labelled with the size itself. sharex strips the labels from
        # inner axes, so only a bottom-row panel carries them.
        assert [t.get_text() for t in axes[-1].get_xticklabels()] == \
            [str(N) for N in SIZES]
        assert len({ax.get_ylim() for ax in axes}) == 1        # sharey
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_interval_grid_by_size_gives_a_block_per_size(tmp_path):
    path = _write_intervals(tmp_path)
    fig, axes = plot_interval_grid(path, n_show=R, group_by="size")
    try:
        # one block per size, R panels each; the pad-out panels are not returned
        assert len(axes) == R * len(SIZES)
        # panels are titled with the replication index, restarting for each size
        assert [ax.get_title() for ax in axes] == \
            [str(i + 1) for i in range(R)] * len(SIZES)
        # sharey spans blocks, not just panels within one block
        assert len({ax.get_ylim() for ax in axes}) == 1
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_interval_grid_by_size_filename_is_distinct(tmp_path):
    """The two layouts must not overwrite each other's figure."""
    path = _write_intervals(tmp_path)
    a = plot_interval_grid(path, outdir=str(tmp_path), stamp="S", n_show=R)
    b = plot_interval_grid(path, outdir=str(tmp_path), stamp="S", n_show=R,
                           group_by="size")
    assert a[0].endswith("_intervals95_grid.png")
    assert b[0].endswith("_intervals95_grid_by_N.png")
    assert a[0] != b[0]


def test_interval_grid_rejects_an_unknown_group_by(tmp_path):
    path = _write_intervals(tmp_path)
    with pytest.raises(ValueError, match="group_by"):
        plot_interval_grid(path, group_by="nonsense")


def test_interval_grid_accepts_a_subset_of_sizes(tmp_path):
    path = _write_intervals(tmp_path)
    fig, axes = plot_interval_grid(path, sizes=[8], n_show=R, group_by="size")
    try:
        assert len(axes) == R          # one block only
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_interval_grid_rejects_an_absent_size(tmp_path):
    path = _write_intervals(tmp_path)
    with pytest.raises(ValueError, match="99"):
        plot_interval_grid(path, sizes=99)


@pytest.mark.parametrize("group_by", ["replication", "size"])
def test_interval_grid_without_sharey_lets_panels_autoscale(tmp_path, group_by):
    """sharey=False must actually reach the subplots call, in both layouts."""
    path = _write_intervals(tmp_path)
    fig, axes = plot_interval_grid(path, n_show=R, sharey=False, group_by=group_by)
    try:
        assert len({ax.get_ylim() for ax in axes}) > 1
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_latest_run_dir_skips_runs_without_the_intervals_file(tmp_path):
    """The real tree has a coverage run predating the endpoint writer, so newest-stamp
    alone is not enough -- the file must be present."""
    (tmp_path / "pyproject.toml").write_text("")
    driver = tmp_path / "scripts" / "ex" / "driver.py"
    driver.parent.mkdir(parents=True)
    driver.write_text("")

    old = tmp_path / "results" / "ex" / "coverage" / "2026-01-01T00-00-00"
    new = tmp_path / "results" / "ex" / "coverage" / "2026-02-01T00-00-00"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / INTERVALS_JSON).write_text("{}")       # only the OLDER run has one

    assert latest_run_dir(str(driver), "coverage",
                          contains=INTERVALS_JSON) == str(old)
    # unfiltered, the newest stamp wins
    assert latest_run_dir(str(driver), "coverage") == str(new)
    # a study that was never run yields None rather than raising
    assert latest_run_dir(str(driver), "limit_theorem") is None
