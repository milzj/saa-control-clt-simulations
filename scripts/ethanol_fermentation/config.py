"""Ethanol fermentation run configuration.

Re-exports the shared study configuration (:mod:`saa_clt.config`) plus the
interior warm-start / feasible-ramp / integration-density helpers, and adds the
endpoint volume cap specific to this example, so the drivers here import
everything from one local ``config`` module.
"""

from saa_clt.config import *          # noqa: F401,F403  (shared seeds/sizes/tols/levels + ipopt_options)
from saa_clt.warmstart import (       # noqa: F401
    project_interior, feasible_ramp, substeps_for)

# Endpoint volume constraint x4(t_f) <= CAP (Banga et al. 2005, Case study II),
# enforced by every SAA solve in all four drivers.
CAP = 200.0
