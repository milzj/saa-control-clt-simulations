"""Shared run configuration for the scenario-based optimal-control studies.

Single source of truth for the knobs every driver shares -- sample sizes,
replication counts, RNG seeds, control-mesh sizes, confidence levels, and the
Ipopt solver tolerances/options -- so they cannot drift between the two examples
(fed-batch reactor and ethanol fermentation).  The two examples import these
verbatim; the ethanol example additionally layers its endpoint volume cap and
integration-density helpers on top (see ``scripts/ethanol_fermentation/config.py``
and :mod:`saa_clt.warmstart`).

The one configuration concern kept elsewhere is the scenario perturbation radius
r, which lives on each model (``ControlProblem.perturbation_radius``).

These values reproduce the settings used for the paper's figures and tables --
do not re-tune them.
"""

import numpy as np

__all__ = [
    "MASTER_SEED",
    "SAA_SAMPLER_SEED", "CLT_ROOT_SEED", "INFERENCE_SEED",
    "INFERENCE_SUBSAMPLE_SEED", "COVERAGE_ROOT_SEED",
    "SOLUTION_NINTERVALS", "SAA_SAMPLES", "SAMPLE_SIZES",
    "CLT_N_REF", "CLT_REPLICATIONS", "CLT_NINTERVALS",
    "INFERENCE_NINTERVALS", "CONFIDENCE_LEVELS",
    "COVERAGE_REPLICATIONS", "COVERAGE_N_REF", "COVERAGE_NINTERVALS",
    "COVERAGE_LEVELS",
    "TOL_SOLUTION", "TOL_INFERENCE",
    "ipopt_options",
]

# ---------------------------------------------------------------------------
# Master seed.  One entropy source for the whole study set; each study draws
# from an INDEPENDENT child stream (numpy SeedSequence.spawn), so the SAA
# ensemble, the CLT replicate draws, the CI anchor draw, and the CI subsample
# index draws never share randomness.  Change MASTER_SEED to re-randomise every
# study at once.  Shared by both examples (same MASTER_SEED and spawn order), so
# the two draw identical RNG streams -- the models differ, so the realised
# scenarios still differ.
# ---------------------------------------------------------------------------
MASTER_SEED = 1234

_saa_ss, _clt_ss, _ci_ss, _ci_sub_ss, _cov_ss = \
    np.random.SeedSequence(MASTER_SEED).spawn(5)


def _seed(seed_sequence):
    """One independent 32-bit int seed per spawned child stream."""
    return int(seed_sequence.generate_state(1)[0])


SAA_SAMPLER_SEED         = _seed(_saa_ss)      # risk-neutral ensemble draw
CLT_ROOT_SEED            = _seed(_clt_ss)      # CLT root (re-spawned into ref + R reps per N)
INFERENCE_SEED           = _seed(_ci_ss)       # CI anchor ensemble draw
INFERENCE_SUBSAMPLE_SEED = _seed(_ci_sub_ss)   # CI subsample index draws (--seed-sub default)
COVERAGE_ROOT_SEED       = _seed(_cov_ss)      # coverage-study root (independent of CLT/CI)

# -- Nominal / risk-neutral (SAA) solution -----------------------------------
SOLUTION_NINTERVALS = 2000     # fine control mesh for the solution demo
SAA_SAMPLES = 128              # risk-neutral ensemble size N (absolute)

# -- Sample-size sweep, shared by the CLT and CI studies ---------------------
SAMPLE_SIZES = (8, 16, 32, 64)   # N in the studies; last entry = subsampling anchor

# -- CLT replication study ---------------------------------------------------
CLT_N_REF = 4096               # reference sample size (proxies J*)
CLT_REPLICATIONS = 300         # replicate SAA solves per N
CLT_NINTERVALS = 50            # coarse control mesh for the repeated solves

# -- Confidence intervals (plug-in + subsampling) ----------------------------
INFERENCE_NINTERVALS = 50      # coarse control mesh for the repeated solves
# Nominal confidence (reliability) levels 1 - beta for the plug-in / subsampling
# CIs.  The coverage study (COVERAGE_LEVELS below) validates exactly these
# levels, so the CIs and their coverage test share ONE definition.
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)

# -- Coverage test of the plug-in CI -----------------------------------------
COVERAGE_REPLICATIONS = 10000    # plug-in replications per N (cost R + 1 solves per N)
COVERAGE_N_REF = CLT_N_REF     # reference sample size proxying J* (shared with the CLT)
COVERAGE_NINTERVALS = CLT_NINTERVALS   # coarse control mesh (matches CLT / inference)
COVERAGE_LEVELS = CONFIDENCE_LEVELS    # coverage validates exactly the CI levels 1 - beta


# -- Ipopt solver options ----------------------------------------------------
# Termination tolerances (Ipopt `tol`).
TOL_SOLUTION = 1e-5    # nominal / risk-neutral (SAA) solutions
TOL_INFERENCE = 1e-3   # CLT replicate + inference (plug-in / subsampling) solves


def ipopt_options(tol, hessian="limited-memory"):
    """Option dict for ``SAAProblem(ipopt_options=...)``."""
    return {"tol": tol, "hessian_approximation": hessian}
