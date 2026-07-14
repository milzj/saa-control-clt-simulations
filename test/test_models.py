"""Each example's model instantiates, exposes a consistent interface, and its
right-hand side / objective evaluate to finite values."""

import numpy as np
import pytest

from _saa_helpers import EXAMPLES, make_model


@pytest.mark.parametrize("example", EXAMPLES)
def test_model_interface_and_evaluation(example):
    model = make_model(example, nintervals=8)

    # initial state has the right length and is finite
    x0 = np.asarray(model.parameterized_initial_state(model.nominal_param[0]), float)
    assert x0.shape == (model.nstates,)
    assert np.all(np.isfinite(x0))

    # control bounds are consistent with ncontrols and are ordered lb <= ub
    lb = np.asarray(model.control_bounds[0], float)
    ub = np.asarray(model.control_bounds[1], float)
    assert lb.shape == (model.ncontrols,) and ub.shape == (model.ncontrols,)
    assert np.all(lb <= ub)

    # the right-hand side Function evaluates to a finite state derivative
    f = model.right_hand_side
    u_mid = 0.5 * (lb + ub)
    k = np.asarray(model.nominal_param[0], float)
    xdot = np.asarray(f(x0, u_mid, k)).ravel()
    assert xdot.shape == (model.nstates,)
    assert np.all(np.isfinite(xdot))

    # the terminal objective is a finite scalar
    assert np.isfinite(float(model.final_cost_function(x0)))


@pytest.mark.parametrize("example", EXAMPLES)
def test_scenario_sampler_shape_and_reproducibility(example):
    model = make_model(example, nintervals=8)
    nparam = len(model.nominal_param[0])
    s1 = np.atleast_2d(np.asarray(model.scenario_sampler(seed=0).sample(5), float))
    s2 = np.atleast_2d(np.asarray(model.scenario_sampler(seed=0).sample(5), float))
    assert s1.shape == (5, nparam)
    assert np.allclose(s1, s2)   # same seed -> same scenarios
