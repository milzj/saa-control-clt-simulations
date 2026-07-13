"""Shared helpers for the smoke tests.

Locates the repo, loads each example's ``model.py`` by file path (both examples name
their model module ``model``, so they cannot both be imported by name), and builds a
tiny solve closure mirroring the drivers (capped + feasible-ramp warm-started for
ethanol; midpoint cold start for the fed-batch reactor).
"""

import os
import importlib.util

import numpy as np
from casadi import inf

import ensemblecontrol
from saa_clt.config import ipopt_options, TOL_INFERENCE
from saa_clt.warmstart import feasible_ramp, substeps_for, project_interior

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
EXAMPLES = ("fed_batch_reactor", "ethanol_fermentation")
MODEL_CLASS = {"fed_batch_reactor": "FedBatchReactor",
               "ethanol_fermentation": "EthanolFermentation"}
CAP = 200.0   # ethanol endpoint volume cap

# Silence Ipopt for the (many, tiny) smoke solves; identical numerics.
_QUIET = dict(ipopt_options(TOL_INFERENCE), print_level=0, sb="yes")


def load_model_class(example):
    path = os.path.join(SCRIPTS, example, "model.py")
    spec = importlib.util.spec_from_file_location("model_" + example, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, MODEL_CLASS[example])


def make_model(example, nintervals):
    model = load_model_class(example)()
    model.nintervals = nintervals
    return model


def make_solve(example, model):
    """A tiny ``solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)``
    mirroring the driver of ``example``."""
    lb = float(model.control_bounds[0][0])
    ub = float(model.control_bounds[1][0])

    if example == "ethanol_fermentation":
        substeps = substeps_for(model.nintervals)

        def solve(samples, w0=None, inner_serial=False):
            with ensemblecontrol.build_lock:
                saa = ensemblecontrol.SAAProblem(
                    model, samples, MultipleShooting=False,
                    steps_per_interval=substeps,
                    terminal_constraints=[(0, lambda x: x[3], -inf, CAP)],
                    ipopt_options=_QUIET)
                ctrl = (project_interior(saa.control_matrix(w0), lb, ub)
                        if w0 is not None
                        else feasible_ramp(model, CAP).reshape(
                            model.nintervals, model.ncontrols))
                saa.initial_decisions = saa.initial_from_controls(ctrl)
            w_opt, f_opt = saa.solve()
            return saa, w_opt, float(np.ravel(f_opt)[0])
    else:
        mid = 0.5 * (np.asarray(model.control_bounds[0], float)
                     + np.asarray(model.control_bounds[1], float))

        def solve(samples, w0=None, inner_serial=False):
            with ensemblecontrol.build_lock:
                saa = ensemblecontrol.SAAProblem(
                    model, samples, MultipleShooting=False, ipopt_options=_QUIET)
                saa.initial_decisions = saa.initial_from_controls(
                    np.tile(mid, (model.nintervals, 1)))
            w_opt, f_opt = saa.solve()
            return saa, w_opt, float(np.ravel(f_opt)[0])

    return solve
