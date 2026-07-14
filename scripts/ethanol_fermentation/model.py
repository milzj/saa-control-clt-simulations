import ensemblecontrol
from casadi import *
import numpy as np

class EthanolFermentation(ensemblecontrol.ControlProblem):
    # Fed-batch fermentation model for ethanol production.
    # Banga et al., "Dynamic optimization of bioprocesses", J. Biotechnol.
    # 117 (2005) 407-419, Case study II (Eqs. 27-36); original model from
    # Cheng & Hwang, Chem. Eng. Commun. 97 (1990) 9-26.
    # States x0=x1 (cell mass), x1=x2 (substrate), x2=x3 (ethanol/product),
    # x3=x4 (volume); control u = feed rate. Dynamics Eqs. (28)-(31), growth
    # rates Eqs. (32)-(33), data Eq. (34)-(35).
    #
    # Deviation from the paper (deliberate): we fix the final time (the paper's
    # problem has a free tf). The endpoint volume constraint x4(tf) <= 200 IS
    # enforced by the drivers via SAAProblem(terminal_constraints=...): since
    # x4' = u is parameter-independent, x4(tf) is identical across the ensemble,
    # so a single-member cap (0, lambda x: x[3], -inf, 200) constrains all
    # scenarios. The model, dynamics and bang-singular control structure are
    # faithful; values differ from the paper's free-tf result
    # (J = x3(tf)*x4(tf) ~ 20839, tf ~ 61.17 h) because tf is fixed.
    def __init__(self):

        super().__init__()

        self._alpha = 0.0            # Mayer-only problem; no running cost (L = 0)
        self._nintervals = 2000       # moderate grid keeps the RK4 substrate/volume
                                     # integration stable over the ~61 h horizon;
                                     # raise this if the dynamics misbehave.
        self._final_time = 61.17     # fixed surrogate for the paper's free tf
                                     # (Banga et al. 2005 report tf = 61.17 h)
        self._ncontrols = 1
        self._nstates = 4

        self._control_bounds = [[0.0], [12.0]]   # 0 <= u <= 12  (Eq. (35))

        self.u = MX.sym("u", 1)
        self.x = MX.sym("x", 4)
        self.L = self.alpha/2*self.u**2
        # k0=0.408, k1=16 (x3 divisor in g1), k2=0.22, k3=71.5 (x3 divisor in g2),
        # k4=0.44, k5=150 (feed substrate conc.), k6=10 (stoichiometric factor).
        self._nominal_param = [[0.408, 16.0, 0.22, 71.5, 0.44, 150.0, 10.0]]
        # Single source of truth for the risk-neutral scenario spread: every
        # driver draws its samples via scenario_sampler(), so this radius cannot
        # drift between demos (see scenario_sampler below).
        self._perturbation_radius = 0.06
        self.params = MX.sym("k", len(self._nominal_param[0]))

    @property
    def control_bounds(self):
        # lower and upper bounds
        return self._control_bounds

    @property
    def nominal_param(self):
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
        # xi_j = (1 + r * U[-1, 1]) * nominal_j with r = perturbation_radius,
        # perturbing the 5 kinetic constants k0..k4 and freezing k5 (feed conc.)
        # and k6 (stoichiometry) via frozen=(5, 6). Centralizing the nominal,
        # radius, and frozen set here keeps every driver on the same spread.
        return ensemblecontrol.UniformRelativeSampler(
            self.nominal_param[0], radius=self.perturbation_radius,
            method=method, seed=seed, frozen=(5, 6), **kwargs)

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

        # x0=x1 (cell mass), x1=x2 (substrate), x2=x3 (ethanol), x3=x4 (volume)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        # Growth rate functions (Eqs. 32-33).
        g1 = (k[0]/(1 + x3/k[1]))*(x2/(k[2] + x2))
        g2 = (1/(1 + x3/k[3]))*(x2/(k[4] + x2))

        xdot = vertcat(\
                        g1*x1 - u*x1/x4,                  # x1' (Eq. 28)
                        -k[6]*g1*x1 + u*(k[5] - x2)/x4,   # x2' (Eq. 29)
                        g2*x1 - u*x3/x4,                  # x3' (Eq. 30)
                        u                                 # x4' (Eq. 31)
                        )
        self.xdot = xdot
        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value  (Eq. (34): x1=1, x2=150, x3=0, x4=10)
        return [1.0, 150.0, 0.0, 10.0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        # Minimize -x3(tf)*x4(tf) (Eq. (27), J = x3(tf)*x4(tf)); maximizes the
        # total amount of ethanol (concentration x volume).
        return -x[2]*x[3]
