"""End-to-end smoke of the study pipeline on tiny problem sizes.

For each example, solve a few small SAA problems and run the plug-in confidence-interval
and central-limit-theorem study routines from ``ensemblecontrol``, using the same solve
closure the drivers use (capped + ramp-warmstarted for ethanol).  These assert the
pipeline runs and produces well-formed output; they do NOT assert the full-precision
manuscript numbers.
"""

import numpy as np
import pytest

import ensemblecontrol

from _saa_helpers import EXAMPLES, make_model, make_solve

SIZES = (2, 4)
# The CLT/inference control mesh (config.CLT_NINTERVALS); coarser meshes make the
# fed-batch single-shooting integration blow up (u/x5 -> NaN).
NINTERVALS = 50


@pytest.mark.parametrize("example", EXAMPLES)
def test_plugin_ci_smoke(example, tmp_path):
    model = make_model(example, nintervals=NINTERVALS)
    solve = make_solve(example, model)
    samples = model.scenario_sampler(seed=0).sample(SIZES[-1])

    solves = ensemblecontrol.solve_saa_prefixes(solve, samples, SIZES)
    records = ensemblecontrol.plugin_sweep(solves, SIZES, levels=(0.95,))

    assert len(records) == len(SIZES)
    band = records[-1]["ci"]["levels"][0.95]
    assert np.isfinite(band["lo"]) and np.isfinite(band["hi"])
    assert band["lo"] <= band["hi"]

    out = tmp_path / "plugin.json"
    ensemblecontrol.save_plugin_run(records, str(out),
                                    r=model.perturbation_radius, meta={})
    assert out.exists()


@pytest.mark.parametrize("example", EXAMPLES)
def test_clt_study_smoke(example, tmp_path):
    model = make_model(example, nintervals=NINTERVALS)
    solve = make_solve(example, model)
    root = model.scenario_sampler(seed=0)

    # mirror the drivers: fed-batch uses a fixed cold start (warm_start=False);
    # ethanol uses the study default (reference-warm-started replicates).
    kwargs = {} if example == "ethanol_fermentation" else {"warm_start": False}
    study = ensemblecontrol.clt_replication_study(
        root, solve, sample_sizes=(4,), R=2, n_ref=6, workers=1, **kwargs)

    assert np.isfinite(study["f_ref"])
    assert set(study["values_by_N"].keys()) == {4}
    assert len(study["values_by_N"][4]) == 2
    assert all(np.isfinite(v) for v in study["values_by_N"][4])
