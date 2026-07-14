"""The local CLT plotter: renders both the fitted normal and the theoretical
N(0, sigma_hat_ref^2) overlay from a saved run, and degrades gracefully on runs
saved before sigma_ref was captured.  Pure plotting -- no solver, no ensemblecontrol
study -- so this stays in the fast smoke suite.
"""

import json

import numpy as np
import pytest

from saa_clt.clt_plots import plot_clt_normal, VARIANTS

SIZES = (8, 16)
SIGMA_REF = 3.5


def _write_run(tmp_path, sigma_ref=SIGMA_REF, f_ref=-32.0):
    """A synthetic clt.json shaped exactly like ensemblecontrol.save_clt_run's."""
    rng = np.random.default_rng(0)
    meta = {"model": "Synthetic", "sampler": "UniformRelativeSampler",
            "sigma": 0.04, "seed": 1234}   # meta["sigma"] = radius, not a variance
    if sigma_ref is not None:
        meta["sigma_ref"] = sigma_ref
    data = {
        "algorithm": "clt",
        "N_ref": 4096,
        "f_ref": f_ref,
        "q": 50,
        "r": 0.04,
        # values are RAW SAA optima; the statistic sqrt(N)*(value - f_ref) is
        # recomputed at plot time, so centre them on f_ref.
        "results": [{"N": N,
                     "values": (f_ref + rng.normal(0, SIGMA_REF / np.sqrt(N), 40)
                                ).tolist()}
                    for N in SIZES],
        "meta": meta,
    }
    path = tmp_path / "clt.json"
    path.write_text(json.dumps(data))
    return path


def test_plot_clt_normal_writes_figures(tmp_path):
    path = _write_run(tmp_path)
    written = plot_clt_normal(str(path), outdir=str(tmp_path), stamp="STAMP")

    for N in SIZES:
        for ext in ("png", "pdf"):
            assert (tmp_path / ("clt_STAMP_clt_N%d.%s" % (N, ext))).exists()
    for ext in ("png", "pdf"):
        assert (tmp_path / ("clt_STAMP_clt_all.%s" % ext)).exists()
    # 3 variants per N, plus the combined overview, in 2 formats
    assert len(written) == (len(SIZES) * len(VARIANTS) + 1) * 2


def test_plot_clt_normal_without_sigma_ref_still_renders(tmp_path):
    # Runs saved before sigma_ref existed must not crash -- fitted curve only, and the
    # theory variant is dropped rather than emitting a bare histogram.
    path = _write_run(tmp_path, sigma_ref=None)
    with pytest.warns(UserWarning, match="sigma_ref"):
        written = plot_clt_normal(str(path), outdir=str(tmp_path), stamp="STAMP")
    assert (tmp_path / "clt_STAMP_clt_all.png").exists()
    for N in SIZES:
        assert not (tmp_path / ("clt_STAMP_clt_N%d_theory.png" % N)).exists()
    assert len(written) == (len(SIZES) * 2 + 1) * 2   # both + fit only


def test_sigma_ref_argument_overrides_meta(tmp_path):
    path = _write_run(tmp_path, sigma_ref=None)
    written = plot_clt_normal(str(path), outdir=str(tmp_path), stamp="STAMP",
                              sigma_ref=SIGMA_REF)   # no warning: explicitly given
    assert (tmp_path / "clt_STAMP_clt_all.png").exists()
    assert len(written) == (len(SIZES) * len(VARIANTS) + 1) * 2


def test_all_three_variants_emit_distinct_files(tmp_path):
    path = _write_run(tmp_path)
    plot_clt_normal(str(path), outdir=str(tmp_path), stamp="STAMP")
    for N in SIZES:
        assert (tmp_path / ("clt_STAMP_clt_N%d.png" % N)).exists()        # both
        assert (tmp_path / ("clt_STAMP_clt_N%d_fit.png" % N)).exists()    # fit only
        assert (tmp_path / ("clt_STAMP_clt_N%d_theory.png" % N)).exists()  # theory


def test_variants_subset_only_emits_requested(tmp_path):
    path = _write_run(tmp_path)
    written = plot_clt_normal(str(path), outdir=str(tmp_path), stamp="STAMP",
                              variants=("theory",), formats=("png",))
    for N in SIZES:
        assert (tmp_path / ("clt_STAMP_clt_N%d_theory.png" % N)).exists()
        assert not (tmp_path / ("clt_STAMP_clt_N%d_fit.png" % N)).exists()
    assert len(written) == len(SIZES) + 1   # + the combined overview


def test_unknown_variant_rejected(tmp_path):
    path = _write_run(tmp_path)
    with pytest.raises(ValueError, match="unknown variant"):
        plot_clt_normal(str(path), outdir=str(tmp_path), stamp="STAMP",
                        variants=("bogus",))


def test_variants_default_to_all_three(tmp_path):
    # The drivers rely on this default: one call must emit all three types per N.
    path = _write_run(tmp_path)
    written = plot_clt_normal(str(path), outdir=str(tmp_path), stamp="STAMP",
                              formats=("png",))
    assert len(written) == len(SIZES) * len(VARIANTS) + 1
