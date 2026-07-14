"""Import smoke: the shared package, the pinned ensemblecontrol release, and the
shared configuration constants (the settings used for the manuscript)."""

import ensemblecontrol

import saa_clt
from saa_clt import config


def test_saa_clt_public_helpers():
    assert callable(saa_clt.ipopt_options)
    assert callable(saa_clt.project_interior)
    assert callable(saa_clt.feasible_ramp)
    assert callable(saa_clt.substeps_for)
    assert callable(saa_clt.study_dir)
    for fn in ("plot_phi_constructed_control", "plot_postprocessed_controls",
               "plot_direct_controls"):
        assert hasattr(saa_clt, fn)


def test_config_constants():
    assert config.MASTER_SEED == 1234
    assert config.SAMPLE_SIZES == (8, 16, 32, 64)
    assert config.COVERAGE_SAMPLE_SIZES == (8, 16, 32, 64)
    assert config.SOLUTION_NINTERVALS == 2000
    assert config.SAA_SAMPLES == 128
    assert config.CLT_N_REF == 4096
    assert config.CLT_REPLICATIONS == 300
    assert config.CLT_NINTERVALS == 50
    assert config.INFERENCE_NINTERVALS == 50
    assert config.CONFIDENCE_LEVELS == (0.90, 0.95, 0.99)
    assert config.TOL_SOLUTION == 1e-5
    assert config.TOL_INFERENCE == 1e-3
    assert config.ipopt_options(1e-3) == {
        "tol": 1e-3, "hessian_approximation": "limited-memory"}


def test_five_independent_seeds():
    seeds = {config.SAA_SAMPLER_SEED, config.CLT_ROOT_SEED, config.INFERENCE_SEED,
             config.INFERENCE_SUBSAMPLE_SEED, config.COVERAGE_ROOT_SEED}
    assert len(seeds) == 5   # SeedSequence(1234).spawn(5), all distinct


def test_ensemblecontrol_public_api():
    # the drivers depend on these public names from the pinned release
    assert ensemblecontrol.build_lock is not None
    assert isinstance(ensemblecontrol.core_budget(), int)
    assert ensemblecontrol.core_budget() >= 1
