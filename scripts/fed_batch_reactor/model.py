import ensemblecontrol
from casadi import *
import numpy as np

class FedBatchReactor(ensemblecontrol.ControlProblem):
    # Based on https://doi.org/10.1109/9.173155
    # and section 5 in https://doi.org/10.1201/9780429123641
    def __init__(self):

        super().__init__()

        self._alpha = 0.0
        self._nintervals = 50
        self._final_time = 15.
        self._ncontrols = 1
        self._nstates = 5

        #TODO: Increase upper bound to 2
        self._control_bounds = [[0], [2]]

        self.u = MX.sym("u", 1)
        self.x = MX.sym("h", 5)
        self.L = self.alpha/2*self.u**2
        # k[0..7] are the ODE reaction parameters; k[8..10] are the nonzero initial
        # states x3(0)=1, x4(0)=5, x5(0)=1, appended so scenario_sampler() perturbs
        # them too -- each scenario then starts from a randomized initial state (read
        # back in parameterized_initial_state). right_hand_side uses only k[0..7].
        self._nominal_param = [[4.75, 0.12, -5, 0.1, 21.87, 0.4, 62.5, -7.3,
                                1.0, 5.0, 1.0]]
        # Single source of truth for the risk-neutral scenario spread: every
        # driver draws its samples via scenario_sampler(), so this radius cannot
        # drift between demos (see scenario_sampler below).
        self._perturbation_radius = 0.04
        self.params = MX.sym("k", len(self._nominal_param[0]))

    @property
    def control_bounds(self):
        # lower and upper bounds
        return self._control_bounds

    @property
    def nominal_param(self):
        # lower and upper bounds
        return self._nominal_param

    @property
    def perturbation_radius(self):
        # relative multiplicative scenario spread r for the risk-neutral SAA
        return self._perturbation_radius

    @perturbation_radius.setter
    def perturbation_radius(self, value):
        self._perturbation_radius = value

    def scenario_sampler(self, seed=None, method="mc", **kwargs):
        # Canonical i.i.d. scenario sampler for the risk-neutral problem:
        # xi_j = (1 + r * U[-1, 1]) * nominal_j with r = perturbation_radius and
        # all parameters perturbed (frozen=()). Centralizing the nominal, radius,
        # and frozen set here keeps every driver on the same spread.
        return ensemblecontrol.UniformRelativeSampler(
            self.nominal_param[0], radius=self.perturbation_radius,
            method=method, seed=seed, frozen=(), **kwargs)

    @property
    def control(self):
        return self.u

    @property
    def state(self):
        return self.x

    @property
    def right_hand_side(self):

        x = self.x
        u = self.u
        k = self.params
        alpha = self._alpha

        x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]

        g3 = k[4]*x4/(x4+k[5])/(x4+k[6])
        g2 = x4*exp(k[2]*x4)/(k[3]+x4)
        g1 = k[0]*g3/(k[1]+g3)

        xdot = vertcat(\
                        g1*(x2-x1)-u*x1/x5,
                        g2*x3-u/x5*x2,
                        g3*x3-u/x5*x3,
                        k[7]*g3*x3+u/x5*(20.-x4),
                        u
                        )
        self.xdot = xdot
        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # x1(0) = x2(0) = 0 are fixed; the nonzero initial states x3(0), x4(0), x5(0)
        # are the random parameters k[8..10], so each scenario starts from a perturbed
        # state ((1 +- r) * nominal (1, 5, 1) via scenario_sampler).
        return [0.0, 0.0, params[8], params[9], params[10]]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return -x[4]*x[0]

