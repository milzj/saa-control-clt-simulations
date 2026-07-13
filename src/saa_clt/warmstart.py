"""Warm-start and integration-density helpers shared by the drivers.

``project_interior`` is used by both examples (the solved controls are
bang-singular-bang, so their bang entries sit on the bounds; an interior-point
warm start needs them strictly inside the box).  ``feasible_ramp`` and
``substeps_for`` / ``INTEGRATION_STEPS`` are used only by the ethanol example
(the endpoint volume cap and the stiff ~61 h dynamics), but they are generic and
kept here so the example configs stay thin.
"""

import numpy as np

__all__ = ["project_interior", "INTEGRATION_STEPS", "substeps_for", "feasible_ramp"]


def project_interior(controls, lb, ub, frac=0.01):
    """Clip controls into the strictly-interior box
    ``[lb + frac*(ub-lb), ub - frac*(ub-lb)]`` for an Ipopt (interior-point) warm
    start.

    The solved control is bang-singular-bang, so its bang entries sit exactly on
    the bounds; feeding those verbatim gives Ipopt a boundary-infeasible start.
    A range-relative margin keeps the guess strictly interior even when
    ``lb = 0``.
    """
    margin = frac * (ub - lb)
    return np.clip(np.asarray(controls, dtype=float), lb + margin, ub - margin)


INTEGRATION_STEPS = 1600   # validated total RK4 steps over the ~61 h horizon (50 x 32)


def substeps_for(nintervals, total=INTEGRATION_STEPS):
    """RK4 substeps per control interval to hold a roughly constant integration
    density (~``total`` steps over the horizon), regardless of the control grid.

    The stiff ethanol dynamics need ~1600 steps or the RK4 truncation error
    inflates x3(t_f) and Ipopt games J = x3(t_f)*x4(t_f).  A coarse 50-interval
    grid (CLT / inference) needs 32 substeps; a fine 2000-interval grid (the
    solution demo) already provides enough integration with 1 -- so the substeps
    track the mesh instead of being hardcoded at 32 everywhere.
    """
    return max(1, -(-int(total) // int(nintervals)))   # ceil(total / nintervals)


def feasible_ramp(model, cap, frac=0.98):
    """A FEASIBLE warm-start control for the volume-capped ethanol solve.

    The cap is a control budget: x4(t_f) = x4(0) + h*sum_k u_k, so
    x4(t_f) <= cap is exactly sum_k u_k <= (cap - x4(0))/h.  A rising ramp
    (1 -> 10 -> 0, which avoids the flat-objective stall of a constant start)
    scaled to sit ``frac`` inside that budget starts feasible near the active
    cap.  The default Ipopt start (u at the box midpoint) violates the cap, so
    cold-started solves need this.
    """
    N = model.nintervals
    h = model.final_time / N
    x40 = float(model.parameterized_initial_state(model.nominal_param[0])[3])
    lb_u, ub_u = model.control_bounds[0][0], model.control_bounds[1][0]
    ramp = np.interp(np.arange(N), [0, 0.9 * N, N - 1], [1.0, 10.0, 0.0])
    budget = frac * (cap - x40) / h
    return np.clip(ramp * (budget / ramp.sum()), lb_u, ub_u)
