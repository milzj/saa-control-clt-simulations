# Fed-batch ethanol fermentation under uncertainty

Based on the fed-batch ethanol fermentation problem in Case Study II of Banga et al.
(2005), [doi:10.1016/j.jbiotec.2005.02.013](https://doi.org/10.1016/j.jbiotec.2005.02.013).
The original problem is deterministic and has a free final time; here we fix the final
time at $t_f = 61.17$ h (the value reported for the best solution in Banga et al.) and
introduce parametric uncertainty in the dynamics.

The state is $x(t,\xi)\in\mathbb{R}^4$, where $x_1$, $x_2$, $x_3$ denote the cell mass,
substrate, and ethanol concentrations, respectively, and $x_4$ denotes the reactor
volume. The scalar control $u(t)$ is the feed rate. For a parameter
$\xi\in\mathbb{R}^5$ the dynamics are

$$
\begin{aligned}
\dot x_1(t,\xi) &= g_1(x(t,\xi),\xi)\,x_1(t,\xi) - \frac{u(t)}{x_4(t)}x_1(t,\xi),\\
\dot x_2(t,\xi) &= -10\,g_1(x(t,\xi),\xi)\,x_1(t,\xi)
                   + \frac{u(t)}{x_4(t)}\bigl(150 - x_2(t,\xi)\bigr),\\
\dot x_3(t,\xi) &= g_2(x(t,\xi),\xi)\,x_1(t,\xi) - \frac{u(t)}{x_4(t)}x_3(t,\xi),\\
\dot x_4(t)     &= u(t),
\end{aligned}
$$

with growth-rate functions

$$
g_1(x,\xi) = \frac{\xi_1}{1 + x_3/\xi_2}\,\frac{x_2}{\xi_3 + x_2},\qquad
g_2(x,\xi) = \frac{1}{1 + x_3/\xi_4}\,\frac{x_2}{\xi_5 + x_2}.
$$

The initial condition is $x_0(\xi) = (1,\,150,\,0,\,10)$, and the control constraints are
$0\le u(t)\le 12$ for $t\in[0,t_f]$ together with the endpoint volume constraint

$$
x_4(t_f)\le 200.
$$

Because $\dot x_4 = u$ is parameter-independent, $x_4(t_f)$ is identical across the
scenario ensemble, so the endpoint cap acts as a single-member terminal constraint
(a control-space budget) enforced by every SAA solve.

The performance index is the total amount of ethanol,

$$
F(x(t_f,\xi)) = -\,x_3(t_f,\xi)\,x_4(t_f,\xi).
$$

## Uncertainty model

The nominal parameter vector is

$$
\bar\xi = (0.408,\;16,\;0.22,\;71.5,\;0.44),
$$

and parametric uncertainty is modeled by relative perturbations of the nominal
parameters,

$$
\xi_j = (1 + r\,\eta_j)\,\bar\xi_j,\qquad j=1,\dots,5,\qquad r = 0.06,
$$

where the $\eta_j$ are independent and uniformly distributed on $[-1,1]$. (In the
implementation the feed substrate concentration $150$ and the stoichiometric factor
$10$ appear as two additional, non-perturbed parameters.)

## Running

From the repository root (with the environment installed, see the top-level
[README](../README.md)):

```bash
scripts/ethanol_fermentation/run_all.sh        # all four studies
# or a single study:
scripts/ethanol_fermentation/run_nominal_saa.sh   # (i)   nominal + SAA solution
scripts/ethanol_fermentation/run_clt.sh           # (ii)  central-limit-theorem study
scripts/ethanol_fermentation/run_inference.sh     # (iii) plug-in + subsampling CIs
scripts/ethanol_fermentation/run_coverage.sh      # (iv)  plug-in coverage (SLOW)
```

Results are written to `results/ethanol_fermentation/<study>/<stamp>/` and run logs
(with timings) to `logs/ethanol_fermentation/`.
