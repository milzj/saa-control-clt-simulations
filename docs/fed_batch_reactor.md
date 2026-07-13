# Fed-batch reactor under uncertainty

Based on the fed-batch reactor control problem of Luus (1992),
[doi:10.1109/9.173155](https://doi.org/10.1109/9.173155). The original problem is
deterministic; here we introduce parametric uncertainty in the numerical constants
appearing in the dynamics and in the initial condition.

The state is $x(t,\xi)\in\mathbb{R}^5$, the scalar feed rate is $u(t)$, and the final
time is fixed at $t_f = 15$ h. For a parameter $\xi\in\mathbb{R}^{11}$ the dynamics are

$$
\begin{aligned}
\dot x_1(t,\xi) &= g_1(x_4(t,\xi),\xi)\bigl(x_2(t,\xi)-x_1(t,\xi)\bigr)
                   - \frac{u(t)}{x_5(t,\xi)}x_1(t,\xi),\\
\dot x_2(t,\xi) &= g_2(x_4(t,\xi),\xi)x_3(t,\xi)
                   - \frac{u(t)}{x_5(t,\xi)}x_2(t,\xi),\\
\dot x_3(t,\xi) &= g_3(x_4(t,\xi),\xi)x_3(t,\xi)
                   - \frac{u(t)}{x_5(t,\xi)}x_3(t,\xi),\\
\dot x_4(t,\xi) &= \xi_8\, g_3(x_4(t,\xi),\xi)x_3(t,\xi)
                   + \frac{u(t)}{x_5(t,\xi)}\bigl(20 - x_4(t,\xi)\bigr),\\
\dot x_5(t,\xi) &= u(t),
\end{aligned}
$$

with kinetic functions

$$
g_3(x_4,\xi) = \frac{\xi_5 x_4}{(x_4+\xi_6)(x_4+\xi_7)},\qquad
g_2(x_4,\xi) = \frac{x_4\exp(\xi_3 x_4)}{\xi_4+x_4},\qquad
g_1(x_4,\xi) = \frac{\xi_1 g_3(x_4,\xi)}{\xi_2+g_3(x_4,\xi)}.
$$

The parameterized initial condition is

$$
x_0(\xi) = (0,\,0,\,\xi_9,\,\xi_{10},\,\xi_{11}),
$$

and the control bounds are $0\le u(t)\le 2$ for $t\in[0,t_f]$.

The performance index is the amount of secreted protein at the final time,

$$
F(x(t_f,\xi)) = -\,x_1(t_f,\xi)\,x_5(t_f,\xi).
$$

## Uncertainty model

The nominal parameter vector is

$$
\bar\xi = (4.75,\;0.12,\;-5,\;0.1,\;21.87,\;0.4,\;62.5,\;-7.3,\;1,\;5,\;1),
$$

where the first eight components $(\xi_1,\dots,\xi_8)$ enter the dynamics and the last
three $(\xi_9,\xi_{10},\xi_{11})$ the initial condition. Parametric uncertainty is
modeled by relative perturbations of the nominal parameters,

$$
\xi_j = (1 + r\,\eta_j)\,\bar\xi_j,\qquad j=1,\dots,11,\qquad r = 0.04,
$$

where the $\eta_j$ are independent and uniformly distributed on $[-1,1]$.

## Running

From the repository root (with the environment installed, see the top-level
[README](../README.md)):

```bash
scripts/fed_batch_reactor/run_all.sh        # all four studies
# or a single study:
scripts/fed_batch_reactor/run_nominal_saa.sh   # (i)   nominal + SAA solution
scripts/fed_batch_reactor/run_clt.sh           # (ii)  central-limit-theorem study
scripts/fed_batch_reactor/run_inference.sh     # (iii) plug-in + subsampling CIs
scripts/fed_batch_reactor/run_coverage.sh      # (iv)  plug-in coverage (SLOW)
```

Results are written to `results/fed_batch_reactor/<study>/<stamp>/` and run logs
(with timings) to `logs/fed_batch_reactor/`.
