"""Unit test for ``saa_clt.outputs.save_coverage_intervals``.

Feeds the writer a hand-built synthetic coverage ``study`` dict (the shape returned by
``ensemblecontrol.coverage_study``: ``bounds_by_N[N][level] = {"lo", "hi"}``) so the test
does no solving and is fast/deterministic.  It checks the JSON round-trips the run
parameters, stores lo/hi for every replication aligned with ``levels``, and that the
recovered center J_hat_N* and standard error are self-consistent with the symmetric
plug-in CI construction.
"""

import json

import numpy as np
from scipy.stats import norm

from saa_clt.outputs import save_coverage_intervals

LEVELS = (0.90, 0.95, 0.99)
SIZES = (4, 8)
R = 5
F_REF = -2.0


def _synthetic_study():
    """A coverage study with a known center/se per (N, replication), from which the
    symmetric plug-in endpoints lo/hi are constructed per level."""
    rng = np.random.default_rng(0)
    z = {lvl: float(norm.ppf(1.0 - (1.0 - lvl) / 2.0)) for lvl in LEVELS}
    bounds_by_N, centers, ses = {}, {}, {}
    for N in SIZES:
        jhat = -2.0 + rng.standard_normal(R)      # some straddle F_REF, some miss
        se = 0.1 + rng.random(R)                  # strictly positive
        centers[N], ses[N] = jhat, se
        bounds_by_N[N] = {lvl: {"lo": jhat - z[lvl] * se, "hi": jhat + z[lvl] * se}
                          for lvl in LEVELS}
    study = {"levels": list(LEVELS), "n_ref": 128, "f_ref": F_REF, "R": R,
             "sample_sizes": list(SIZES), "bounds_by_N": bounds_by_N}
    return study, centers, ses, z


def test_save_coverage_intervals(tmp_path):
    study, centers, ses, z = _synthetic_study()
    out = tmp_path / "coverage_plugin_intervals.json"

    ret = save_coverage_intervals(study, str(out), meta={"model": "X", "ci": "plugin"})
    assert ret == str(out)
    assert out.exists()

    data = json.loads(out.read_text())

    # run parameters round-trip
    assert data["algorithm"] == "coverage_intervals"
    assert data["ci"] == "plugin"
    assert data["levels"] == list(LEVELS)
    assert data["n_ref"] == 128
    assert data["f_ref"] == F_REF
    assert data["R"] == R
    assert data["sample_sizes"] == list(SIZES)
    assert data["meta"] == {"model": "X", "ci": "plugin"}

    assert [res["N"] for res in data["results"]] == sorted(SIZES)
    for res in data["results"]:
        N = res["N"]
        lo, hi = np.array(res["lo"]), np.array(res["hi"])
        # one column per level, each of length R (all replications stored)
        assert lo.shape == (len(LEVELS), R)
        assert hi.shape == (len(LEVELS), R)

        jhat = np.array(res["J_hat_N"])
        se = np.array(res["se"])
        assert jhat.shape == (R,) and se.shape == (R,)

        # symmetric CI: center and se recovered exactly from the endpoints, and the
        # stored values match the synthetic ground truth.
        np.testing.assert_allclose(jhat, centers[N])
        np.testing.assert_allclose(se, ses[N])
        for j, lvl in enumerate(LEVELS):
            np.testing.assert_allclose(0.5 * (lo[j] + hi[j]), jhat)
            np.testing.assert_allclose(0.5 * (hi[j] - lo[j]) / z[lvl], se)

        # covers is derivable from the stored endpoints and f_ref
        covers = (lo <= F_REF) & (F_REF <= hi)
        # wider (higher-level) intervals cover at least as often as narrower ones
        assert covers[2].sum() >= covers[1].sum() >= covers[0].sum()
