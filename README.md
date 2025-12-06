# Computational Physics Exam Code Pack

This folder contains ready‑to‑use Python templates for your computational physics coding exam, organized by module and topic.

All codes follow the same basic pattern:

- Pure Python + `math` + optional `matplotlib` (no NumPy/SciPy in most exam-style logic; a few FD templates use NumPy for convenience if allowed).
- Plain lists and for‑loops where appropriate.
- A clearly marked **USER PARAMETERS** section near the top.
- Output to one or more `.dat` files with a commented header (`# col1: ...`).
- Optional **matplotlib** sanity plots if the library is available.
- Printed example **gnuplot** commands at the end.

You can answer almost any exam question by:
1. Picking the right script from the correct module.
2. Editing the *user parameters* and the *problem‑specific functions* (`f(x)`, ODE RHS, matrix A, etc.).
3. Running:
   ```bash
   python3 script_name.py
   ```
4. Using the suggested gnuplot commands to plot from the `.dat` files.

---

## 0. Folder structure

Inside `Codes/` you have these subfolders:

- `Root Finding/`
- `Numerical Integration/`
- `System of Linear Equations/`
- `ODE/`
- `PDE/`
- `Monte Carlo/`
- `Markov Chains/`

Each maps directly to the exam modules:

- **Module 1** → Root Finding  
- **Module 2** → Numerical Integration & Interpolation  
- **Module 3** → Systems of Linear Equations  
- **Module 4** → ODEs (IVP, BVP, shooting, finite difference)  
- **Module 5** → PDEs (Laplace, Heat, Wave, general 1D PDE)  
- **Module 6** → Monte Carlo & Random Numbers  
- **Module 7** → Markov Chains & MCMC

Below is a quick “when to use what” guide.

---

## 1. Module 1 – Root finding

Folder: `Codes/Root Finding/`

Scripts:

- `bissection_method.py`
- `newton_raphson.py`

### 1.1 `bissection_method.py`

Use this when:

- You are asked to solve `f(x) = 0` on a **bracket** `[a, b]` with `f(a)*f(b) < 0`.
- You need **guaranteed convergence** and **simple error tracking**:
  - Script tracks `x_mid` and `|x_n − x_{n-1}|` per iteration.
  - Writes `(iteration, x_n, error_n)` to a `.dat` file.

Edit in the exam:

- USER PARAMETERS:
  - `a`, `b`, `tol`, `max_iter`, `output_file`.
- Function `f(x)` to match the given equation.

Use gnuplot, e.g.:

```gnuplot
plot 'bisection_convergence.dat' using 1:3 with linespoints title 'x_n'
plot 'bisection_convergence.dat' using 1:4 with linespoints title '|x_n - x_{n-1}|'
```

---

### 1.2 `newton_raphson.py`

Use this when:

- You have a **good initial guess** `x0` and (usually) a smooth function.
- The question is about **fast convergence** (Newton’s method).
- You may be asked to use an **analytic derivative** or a **finite‑difference derivative**.

Edit in the exam:

- USER PARAMETERS: `x0`, `tol`, `max_iter`, `output_file`.
- Function `f(x)`.
- Either:
  - Define `df(x)` (analytic derivative), or
  - Use the built‑in finite‑difference approximation (check comments in the file).

Outputs:

- Iteration, current `x_n`, and `|x_n − x_{n-1}|` to a `.dat` file.
- Example gnuplot similar to bisection.

---

## 2. Module 2 – Numerical integration & interpolation

Folder: `Codes/Numerical Integration/`

Scripts:

- `trapezoidal_rule.py`
- `simpson_one_third.py`
- `simpson_three_eighth.py`
- `gauss_legendre_generic.py`
- `gauss-legendre.py` (fixed small orders)
- `newton_forward_interp.py`
- `improper_integration.py`

### 2.1 `trapezoidal_rule.py`

Use this when:

- You need the **composite trapezoidal rule** for `∫_a^b f(x) dx`.
- You must show **convergence in N** using purely numerical error `|I_N − I_prev|`.

Edit:

- USER PARAMETERS: `a`, `b`, `N_list`, `output_file`.
- Function `f(x)`.

The script loops over `N_list`, computes `I_N`, and stores `N, h, I_N, |I_N − I_prev|`.

---

### 2.2 `simpson_one_third.py`

Use this for:

- Simpson’s 1/3 rule with multiple even `N` values.
- Questions asking for **higher‑order accuracy** and **numerical convergence**.

Pattern is the same as your example snippet:
- User sets `a`, `b`, `N_list`, and `f(x)`.
- Output: `simpson13_convergence.dat`.

---

### 2.3 `simpson_three_eighth.py`

Use this when:

- The exam specifically asks for **Simpson’s 3/8 rule**.
- Or when the step count is a multiple of 3.

Edit:

- USER PARAMETERS: `a`, `b`, `N_list`, `output_file`.
- Function `f(x)`.

Behavior:

- Similar to 1/3 rule: for each `N`, computes `I_N` and `error_N = |I_N − I_prev|`.
- Produces `.dat` + optional matplotlib plots.

---

### 2.4 `gauss_legendre_generic.py`

Use this when:

- They ask for **Gauss–Legendre quadrature** with a **general number of points**.
- You need to integrate on `[a, b]` but use standard [-1, 1] nodes internally.

Edit:

- USER PARAMETERS:
  - `a`, `b`
  - `n_points` (2, 3, 4, 5, ...)
  - `output_file`
- Function `f(x)`.

The script:

- Picks nodes/weights for allowed `n_points`.
- Maps `[a, b]` → `[-1, 1]`.
- Writes `n_points, I_n` and optionally compares different orders.

---

### 2.5 `newton_forward_interp.py`

Use this for:

- **Interpolation** of tabulated data (equally spaced x).
- Constructing Newton’s forward interpolation polynomial.
- Showing **error via more points**: compare `m` vs `m+1` points.

Edit:

- USER PARAMETERS:
  - Arrays of `x_data`, `y_data`
  - `x_eval` (where to evaluate).
- The script builds the **forward difference table** and evaluates `P(x)`.

---

### 2.6 `improper_integration.py`

Use this for:

- Improper integrals using substitution to a finite interval:
  - `[a, ∞)`, `(-∞, b]`, `(-∞, ∞)`, and finite `[a, b]`.

Edit:

- `improper_type`: `"FINITE"`, `"A_INF"`, `"MINF_B"`, `"MINF_INF"`.
- Parameters `a`, `b` (used depending on type).
- Function `f(x)`.
- `N_list` for t‑grid refinements.

The script handles all the various substitutions and does Simpson 1/3 in t.

---

## 3. Module 3 – Systems of linear equations

Folder: `Codes/System of Linear Equations/`

Scripts:

- `gaussian_elimination_pivoting.py`
- `jacobi_iteration.py`
- `gauss_seidel_iteration.py`

### 3.1 `gaussian_elimination_pivoting.py`

Use this when:

- You need **direct Gaussian elimination with partial pivoting**.
- Solve a single linear system `A x = b`.

Edit:

- USER SECTION:
  - Define matrix `A` (as a list of lists).
  - Define vector `b`.
- The script:
  - Performs forward elimination with pivoting.
  - Back substitution.
  - Optionally prints residuals.

---

### 3.2 `jacobi_iteration.py`

Use this when:

- The exam asks for **Jacobi iterative method**.
- You must show **convergence behavior** with error `||x^(n) − x^(n-1)||`.

Edit:

- USER SECTION:
  - Matrix `A`, vector `b`.
  - Initial guess.
  - `max_iter`, `tol`.

Output:

- Iteration vs numerical error to `.dat` for plotting.
- Use this to show slow convergence / divergence.

---

### 3.3 `gauss_seidel_iteration.py`

Use this for:

- **Gauss–Seidel** iterative solution of `A x = b`.
- Comparing convergence speed against Jacobi.

Edit:

- Same kind of USER SECTION as Jacobi.

Output:

- Iteration vs `||x^(n) − x^(n-1)||` to `.dat`.

---

## 4. Module 4 – ODEs (IVP, BVP, shooting, finite difference)

Folder: `Codes/ODE/`

Scripts:

- `euler_ode.py`
- `RK2.py`
- `RK3.py`
- `RK4.py`
- `RK_coupled.py`
- `shooting_method.py`
- `finite_difference.py`
- `finite_difference_BVP-IVP.py`

### 4.1 `euler_ode.py`, `RK2.py`, `RK3.py`, `RK4.py`

Use these when:

- You have an IVP: `y' = f(x, y)`, `y(x0) = y0`.
- You need **single‑step time marching** with step size `h`.
- You may need **step‑refinement error** by comparing `h` and `h/2`.

Edit:

- USER PARAMETERS:
  - `x0`, `y0`, `x_end`, `h`.
- Function `f(x, y)`.

`RK_coupled.py` is similar but for **coupled systems** `y1', y2', ...`.

---

### 4.2 `shooting_method.py`

Use this when:

- You have a **2nd order BVP** and you want to solve it via shooting:
  - Convert to a first‑order system.
  - Guess the missing derivative at one boundary.
  - Adjust until the boundary condition at the other end is matched.

Edit:

- USER PARAMETERS: `a`, `b`, boundary values.
- Define the system `y1' = ...`, `y2' = ...`.
- The script runs RK inside and prints error vs shooting parameter.

---

### 4.3 `finite_difference.py`

Use this for:

- Classical **2nd‑order BVP** in FD form with standard Dirichlet BCs.

If you want mixed BC/IC, use the next one.

---

### 4.4 `finite_difference_BVP-IVP.py`

Use this when:

- You need a **unified FD solver** that can handle:
  - BVP: `y(a)=A`, `y(b)=B`.
  - Mixed: `y(a)=A`, `y'(b)=D` or `y'(a)=D`, `y(b)=B`.
  - “IVP‑like” conditions with derivative at one end.

Edit:

- USER PARAMETERS:
  - `a`, `b`, `N`.
  - `left_type`, `right_type` as `"BC"` or `"IC"`.
  - `left_value`, `right_value`.
- Define the ODE in the linear form:
  - `y'' + P(x)*y' + Q(x)*y = R(x)` via functions `P(x)`, `Q(x), R(x)`.

The script:

- Builds FD system `A y = b`.
- Solves it.
- Writes `x, y` to `.dat` and optionally plots `y(x)`.

---

## 5. Module 5 – PDEs (Laplace, Heat, Wave)

Folder: `Codes/PDE/`

Scripts:

- `laplace2d_gauss_seidel.py`
- `heat1d_explicit.py`
- `heat1d_implicit.py`
- `heat1d_crank_nicolson.py`
- `heat2d_explicit.py`
- `heat2d_implicit.py`
- `heat2d_crank_nicolson.py`
- `pde1d_general_explicit.py`
- `pde1d_general_implicit.py`
- `wave1d_central.py`

### 5.1 `laplace2d_gauss_seidel.py`

Use this when:

- You solve **steady‑state Laplace**: `∇²u = 0` on a rectangle.
- With **Dirichlet boundary conditions** on all sides.
- Need **iterative Gauss–Seidel relaxation**.

Edit:

- Grid size `Nx`, `Ny`, physical domain, boundary arrays.

Outputs `(i, j, u_ij)` to file and optionally 2D plots via gnuplot.

---

### 5.2 Heat equation (1D)

- `heat1d_explicit.py` → FTCS scheme (forward time, central space).
- `heat1d_implicit.py` → fully implicit scheme.
- `heat1d_crank_nicolson.py` → Crank–Nicolson scheme.

Use them depending on what scheme the exam mentions.

Edit:

- `alpha` (diffusivity), `dx`, `dt`, `Nx`, `Nt`.
- Initial and boundary conditions in the USER SECTION.

They output `x, t, u(x,t)` snapshots to `.dat` files.

---

### 5.3 Heat equation (2D)

- `heat2d_explicit.py`
- `heat2d_implicit.py`
- `heat2d_crank_nicolson.py`

Use these for 2D heat/diffusion questions.
Edit domain, grid, and initial/boundary conditions.

---

### 5.4 `pde1d_general_explicit.py`, `pde1d_general_implicit.py`

Use these when:

- You have a **general 1D PDE** of the form:
  - `u_t = A(x) u_xx + B(x) u_x + C(x) u + S(x, t)`.
- Need a template where you only change coefficient functions.

Edit coefficient functions and parameters; the scheme (explicit or implicit) is already coded.

---

### 5.5 `wave1d_central.py`

Use this for:

- `u_tt = c^2 u_xx` wave equation in 1D.
- Central differences in both time and space.

Edit:

- `c`, domain, `dx`, `dt`, initial displacement `u(x,0)` and velocity `u_t(x,0)`.

Outputs snapshots for plotting.

---

## 6. Module 6 – Monte Carlo methods

Folder: `Codes/Monte Carlo/`

Scripts:

- `lcg_prng.py`
- `tests_moment_autocorr_ks.py`
- `inverse_transform_exponential.py`
- `box_muller.py`
- `accept_reject.py`
- `crude_integration_1d.py`
- `mc_integration_d_dim.py`
- `importance_sampling.py`
- `importance_sampling_autoinverse.py`
- `variance_antithetic.py`
- `variance_control_variate.py`
- `variance_stratified.py`

### 6.1 `lcg_prng.py`

Use this when:

- You must implement a **Linear Congruential Generator**.
- Generate uniform(0,1) random numbers and write them to `.dat`.

USER SECTION: modulus `m`, multiplier `a`, increment `c`, seed `X0`, number of samples.

---

### 6.2 `tests_moment_autocorr_ks.py`

Use this to:

- Test a sequence of uniforms from the LCG:
  - **k‑th moments**,  
  - **autocorrelation**,  
  - **Kolmogorov–Smirnov** statistic.

You point it to an input sequence or generate inside.

---

### 6.3 Non‑uniform generators

- `inverse_transform_exponential.py` → Inverse CDF for exponential.
- `box_muller.py` → Box–Muller method for standard normal.
- `accept_reject.py` → General accept/reject sampling using input target `f(x)` and envelope.

Use depending on the exam’s request for method of generation.

---

### 6.4 Monte Carlo integration

- `crude_integration_1d.py`:
  - Basic MC integration on [a,b].
  - Outputs estimate, variance, and standard error.
- `mc_integration_d_dim.py`:
  - D‑dimensional crude Monte Carlo integral.

Edit:

- Domain `[a,b]` (or hypercube in D).
- Function `f(x)` or `f(x1, ..., xD)`.
- Number of samples.

---

### 6.5 Importance sampling

- `importance_sampling.py`:
  - Simple version (fixed q(x), usually q=2x on [0,1]).
- `importance_sampling_autoinverse.py`:
  - General version where you can define `f(x)` and **also change `q(x)`**; the script builds a numerical inverse CDF and does all the weight calculations automatically.

Use the **`autoinverse`** version when:

- You want to tune q(x) to be closer to f(x).
- You do not want to hand‑derive the inverse CDF.

---

### 6.6 Variance reduction

- `variance_antithetic.py` → Antithetic variables.
- `variance_control_variate.py` → Control variates.
- `variance_stratified.py` → Stratified sampling.

Use these when exam questions explicitly ask for **variance reduction techniques** and comparison of variance vs crude Monte Carlo.

---

## 7. Module 7 – Markov chains & MCMC

Folder: `Codes/Markov Chains/`

Scripts:

- `1d_metropolois-hastings_algorithm.py`
- `1d_random_walk_metropolis-hastings.py`
- `ising2d_metropolis.py`

### 7.1 `1d_metropolois-hastings_algorithm.py`

Use this for:

- **Independent** Metropolis–Hastings in 1D:
  - Proposal does not depend on current x (e.g., sample from q and accept/reject).
- Sampling from a target density given up to normalization.

Edit:

- Target `f_target_unnorm(x)`.
- Proposal sampler `sample_proposal(rng)` and `q_pdf(x)`.
- Number of steps, burn‑in, etc.

Outputs:

- Chain values, acceptance rate, sample mean, variance, and autocorrelation.

---

### 7.2 `1d_random_walk_metropolis-hastings.py`

Use this for:

- **Random walk** MH:
  - Proposal of the form `x_new = x_old + step * normal(0,1)` (Box–Muller inside).

Edit:

- Target `f_target_unnorm(x)`.
- Step size.
- Number of samples, burn‑in.

Outputs chain + statistics similar to above.

---

### 7.3 `ising2d_metropolis.py`

Use this when:

- The exam asks for a simple 2D **Ising model** with Metropolis updates:
  - Square lattice with spins ±1.
  - Energy difference ΔE for flipping a spin.
  - Boltzmann factor `exp(-ΔE / (kT))`.

Edit:

- Lattice size `L`.
- Temperature `T`.
- Number of sweeps.
- Initial configuration (random or ordered).

Outputs:

- Magnetization vs Monte Carlo steps.
- Optionally energy vs steps.
- Mean and variance of magnetization.
- Autocorrelation of magnetization time series.

---

## 8. General tips during the exam

1. **Always start from the closest template**  
   Don’t reinvent logic. If the question says “use Simpson’s rule” → go straight to the Simpson template and change `f(x)` + parameters.

2. **Use numerical error only**  
   All convergence/error plots are based on differences between:
   - successive iterations (`|x_n − x_{n-1}|`), or
   - successive resolutions (`|I_h − I_{h/2}|`),  
   not exact analytic solutions.

3. **Keep `.dat` headers intact**  
   The commented `# col1: ...` lines are your quick reference for gnuplot.

4. **Matplotlib is only for sanity**  
   If available, you get instant visual checks. In a strict environment, you still have `.dat` + gnuplot.

5. **If stuck, reduce the problem**  
   - For ODEs: start from Euler, then upgrade to RK2/RK4.  
   - For PDEs: start from 1D explicit, then move to implicit / Crank–Nicolson.  
   - For MC: start from crude integration, then add importance sampling or variance reduction.

Good luck – this pack already covers almost everything they can realistically throw at you. Just pick the right template, plug in the question’s data, and you’re set.
