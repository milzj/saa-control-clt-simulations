"""Shared, de-duplicated helpers for the two optimal-control examples.

The two examples (``scripts/fed_batch_reactor`` and
``scripts/ethanol_fermentation``) each contribute only their model and a thin
config; everything they share -- the run configuration, the warm-start helpers,
the control post-processing plots, and the output-path helper -- lives here.
The numerical study routines themselves come from the external ``ensemblecontrol``
package (pinned in ``pyproject.toml``).
"""

from . import config
from . import warmstart
from . import outputs
from . import postprocess
from . import clt_plots
from . import coverage_plotting
from .config import ipopt_options
from .warmstart import (project_interior, feasible_ramp, substeps_for,
                        INTEGRATION_STEPS)
from .outputs import (study_dir, repo_root, example_name, latest_run_dir,
                      save_coverage_intervals, load_coverage_intervals,
                      save_clt_reference)
from .postprocess import (plot_phi_constructed_control,
                          plot_postprocessed_controls, plot_direct_controls)
from .clt_plots import plot_clt_normal
from .coverage_plotting import plot_coverage_intervals, plot_interval_grid

__all__ = [
    "config", "warmstart", "outputs", "postprocess", "clt_plots",
    "coverage_plotting", "ipopt_options",
    "project_interior", "feasible_ramp", "substeps_for", "INTEGRATION_STEPS",
    "study_dir", "repo_root", "example_name", "latest_run_dir",
    "save_coverage_intervals", "load_coverage_intervals", "save_clt_reference",
    "plot_phi_constructed_control", "plot_postprocessed_controls",
    "plot_direct_controls", "plot_clt_normal", "plot_coverage_intervals",
    "plot_interval_grid",
]
