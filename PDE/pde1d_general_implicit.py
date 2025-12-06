"""
General 1D time-dependent PDE – implicit (Backward Euler) finite difference.

PDE form (1D):
    u_t = a(x, t, u) * u_xx
          + b(x, t, u) * u_x
          + c(x, t, u) * u
          + s(x, t, u)

Domain:
    x in [0, L],  t in [0, T]

Grid:
    x_i = i * dx,   i = 0..Nx          with dx = L / Nx
    t_n = n * dt,   n = 0..Nt          with dt = T / Nt

Finite differences (spatial):
    u_i^n      ≈ u(x_i, t_n)
    u_xx^{n+1,i} ≈ (u_{i+1}^{n+1} - 2 u_i^{n+1} + u_{i-1}^{n+1}) / dx^2
    u_x^{n+1,i}  ≈ (u_{i+1}^{n+1} - u_{i-1}^{n+1}) / (2 dx)

Backward Euler in time:
    (u_i^{n+1} - u_i^n) / dt =
         a_i * u_xx^{n+1,i}
       + b_i * u_x^{n+1,i}
       + c_i * u_i^{n+1}
       + s_i

where a_i, b_i, c_i, s_i are evaluated at (x_i, t_n, u_i^n)
for simplicity (this is standard in many exam problems; if coefficients
are constant in x,t,u, this matches the usual formulas).

Rearranging, for interior nodes i = 1..Nx-1:

    (-rdiff_i + radv_i) * u_{i-1}^{n+1}
  + (1 + 2 rdiff_i - rreact_i) * u_i^{n+1}
  + (-rdiff_i - radv_i) * u_{i+1}^{n+1}
  = u_i^n + dt * s_i

where:
    rdiff_i   = dt * a_i / dx^2
    radv_i    = dt * b_i / (2 dx)
    rreact_i  = dt * c_i

This defines a tri-diagonal linear system for the interior values
U^{n+1} = [u_1^{n+1}, ..., u_{Nx-1}^{n+1}]^T at each time step.

Boundary conditions (Dirichlet):
    u(0, t)   = g_left(t)
    u(L, t)   = g_right(t)

Initial condition:
    u(x, 0)   = u_init(x)

----------------------------------------------------------------------
What this script does:
----------------------------------------------------------------------
1. Sets up a 1D grid in space and time.
2. Initializes u(x,0) from u_init(x) and applies Dirichlet boundaries.
3. For each time step:
    - Evaluates a, b, c, s at (x_i, t_n, u_i^n).
    - Builds tri-diagonal coefficients for interior nodes.
    - Adjusts RHS for boundary values at time t_{n+1}.
    - Solves tri-diagonal system with the Thomas algorithm.
    - Forms u^{n+1} including boundaries.
    - Computes purely numerical error:
          E_n = max_i |u_i^{n+1} - u_i^n|.
4. Writes solution to 'pde1d_implicit_solution.dat':
       col1: n       (time index)
       col2: t_n     (time)
       col3: i       (space index)
       col4: x_i
       col5: u_i^n
5. Writes convergence to 'pde1d_implicit_convergence.dat':
       col1: n
       col2: t_n
       col3: E_n = max_i |u_i^n - u_i^{n-1}|

6. Optionally (if matplotlib is available):
    - Plots u(x,T) at final time.
    - Plots E_n vs t_n on a semilog scale, skipping the n=0 point
      so the first “0” error does not clutter the plot.

----------------------------------------------------------------------
How to use in the exam:
----------------------------------------------------------------------
- Step 1: Edit USER PARAMETERS:
      L, T, Nx, Nt and output filenames.
- Step 2: Edit:
      a_coeff, b_coeff, c_coeff, s_source, u_init, g_left, g_right
  to match the given PDE, initial and boundary conditions.
- Step 3: Run:
      python3 pde1d_general_implicit.py
- Step 4: Use gnuplot, e.g.:

    # Final-time profile (last block only: last (Nx+1) lines):
    plot 'pde1d_implicit_solution.dat' using 4:5 every :::(Nt*(Nx+1)):: \
         with linespoints title 'u(x,T)'

    # Convergence vs time (skipping n=0 row):
    plot 'pde1d_implicit_convergence.dat' using 2:3 every ::1 \
         with linespoints title 'E_n'

Note: Error/convergence is purely numerical (difference between time levels).
No analytic solution is used anywhere in the code.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
L  = 1.0          # spatial domain length: x in [0, L]
T  = 0.1          # final time

Nx = 50           # number of spatial intervals (Nx+1 grid points)
Nt = 200          # number of time steps

output_sol  = "pde1d_implicit_solution.dat"
output_conv = "pde1d_implicit_convergence.dat"
# ===============================================================


# ---------- PDE COEFFICIENTS & SOURCE: EDIT FOR YOUR PROBLEM ----------

def a_coeff(x, t, u):
    """
    Diffusion coefficient a(x,t,u) multiplying u_xx.

    Example (heat equation with alpha = 1):
        return 1.0
    """
    return 1.0


def b_coeff(x, t, u):
    """
    Advection coefficient b(x,t,u) multiplying u_x.

    Example:
        return v   # constant advection speed
    """
    return 0.0


def c_coeff(x, t, u):
    """
    Reaction coefficient c(x,t,u) multiplying u.

    Example:
        return -lambda_value
    """
    return 0.0


def s_source(x, t, u):
    """
    Source term s(x,t,u). For no source, return 0.0.
    """
    return 0.0


# ---------- INITIAL & BOUNDARY CONDITIONS: EDIT FOR YOUR PROBLEM ----------

def u_init(x):
    """
    Initial condition u(x, 0).

    Example here: a Gaussian bump.
    """
    return math.exp(-50.0 * (x - 0.5 * L) ** 2)


def g_left(t):
    """
    Dirichlet boundary at x = 0: u(0, t).
    """
    return 0.0


def g_right(t):
    """
    Dirichlet boundary at x = L: u(L, t).
    """
    return 0.0


# ---------- TRI-DIAGONAL SOLVER (Thomas algorithm) ----------

def solve_tridiagonal(a_lower, a_diag, a_upper, d):
    """
    Solve a tri-diagonal linear system A x = d with:
        a_lower[k] x_{k-1} + a_diag[k] x_k + a_upper[k] x_{k+1} = d[k]

    for k = 0..M-1, where:
        - a_lower[0] is unused (can be 0)
        - a_upper[M-1] is unused (can be 0)

    This uses the Thomas algorithm (forward elimination + back substitution).
    Arrays are modified in-place.
    """
    n = len(a_diag)

    # Forward sweep
    for k in range(1, n):
        if abs(a_diag[k - 1]) < 1e-14:
            print("Warning: near-zero pivot in tri-diagonal solver.")
            continue
        m = a_lower[k] / a_diag[k - 1]
        a_diag[k] -= m * a_upper[k - 1]
        d[k]      -= m * d[k - 1]

    # Backward substitution
    x = [0.0 for _ in range(n)]
    if abs(a_diag[n - 1]) < 1e-14:
        print("Warning: near-zero last diagonal in tri-diagonal solver.")
        x[n - 1] = 0.0
    else:
        x[n - 1] = d[n - 1] / a_diag[n - 1]

    for k in range(n - 2, -1, -1):
        if abs(a_diag[k]) < 1e-14:
            print("Warning: near-zero diagonal in back substitution.")
            x[k] = 0.0
        else:
            x[k] = (d[k] - a_upper[k] * x[k + 1]) / a_diag[k]

    return x


# ----------------------------------------------------------------------
# Main implicit solver
# ----------------------------------------------------------------------

def main():
    dx = L / float(Nx)
    dt = T / float(Nt)

    print("General 1D PDE implicit (Backward Euler) solver")
    print(f"  dx = {dx:.3e}, dt = {dt:.3e}")
    print("  PDE: u_t = a u_xx + b u_x + c u + s")
    print("  NOTE: Coefficients a,b,c are evaluated at the old time level.")

    # Spatial grid
    x = [i * dx for i in range(Nx + 1)]

    # Allocate solution arrays
    u_old = [0.0 for _ in range(Nx + 1)]
    u_new = [0.0 for _ in range(Nx + 1)]

    # Initial condition at t=0
    t = 0.0
    for i in range(Nx + 1):
        u_old[i] = u_init(x[i])

    # Apply Dirichlet boundaries at t=0
    u_old[0]  = g_left(t)
    u_old[Nx] = g_right(t)

    # Lists for convergence
    E_list = []
    t_list = []

    # Open solution file and write initial profile
    with open(output_sol, "w") as f_sol:
        f_sol.write("# General 1D PDE implicit (Backward Euler) solution\n")
        f_sol.write("# col1: n (time index)\n")
        f_sol.write("# col2: t_n\n")
        f_sol.write("# col3: i (space index)\n")
        f_sol.write("# col4: x_i\n")
        f_sol.write("# col5: u_i^n\n")

        # n=0 snapshot
        for i in range(Nx + 1):
            f_sol.write(f"0 {t} {i} {x[i]} {u_old[i]}\n")

        E_list.append(0.0)   # no previous step
        t_list.append(t)

        # Time-stepping
        for n in range(1, Nt + 1):
            t = n * dt

            # Dirichlet BC at new time (used in RHS)
            u_new[0]  = g_left(t)
            u_new[Nx] = g_right(t)

            # Tri-diagonal arrays for interior system (nodes i=1..Nx-1)
            M = Nx - 1  # number of interior unknowns
            a_lower = [0.0 for _ in range(M)]  # sub-diagonal
            a_diag  = [0.0 for _ in range(M)]  # main diagonal
            a_upper = [0.0 for _ in range(M)]  # super-diagonal
            d       = [0.0 for _ in range(M)]  # RHS

            # Build system row by row
            for i in range(1, Nx):
                k = i - 1  # interior index 0..M-1

                ui = u_old[i]
                t_old = t - dt

                # Coefficients at old time
                a_val = a_coeff(x[i], t_old, ui)
                b_val = b_coeff(x[i], t_old, ui)
                c_val = c_coeff(x[i], t_old, ui)
                s_val = s_source(x[i], t_old, ui)

                rdiff   = dt * a_val / (dx * dx)
                radv    = dt * b_val / (2.0 * dx)
                rreact  = dt * c_val

                # Tri-diagonal coefficients for u_{i-1}^{n+1}, u_i^{n+1}, u_{i+1}^{n+1}
                coeff_left  = -rdiff + radv
                coeff_diag  = 1.0 + 2.0 * rdiff - rreact
                coeff_right = -rdiff - radv

                # Start RHS with u_i^n + dt * s
                d_k = u_old[i] + dt * s_val

                # Boundary contributions:
                # left neighbor:
                if i == 1:
                    # u_{0}^{n+1} is known boundary at x=0
                    d_k -= coeff_left * u_new[0]
                else:
                    a_lower[k] = coeff_left

                # right neighbor:
                if i == Nx - 1:
                    # u_{Nx}^{n+1} is known boundary at x=L
                    d_k -= coeff_right * u_new[Nx]
                else:
                    a_upper[k] = coeff_right

                a_diag[k] = coeff_diag
                d[k]      = d_k

            # Solve the tri-diagonal system for interior nodes
            U_int = solve_tridiagonal(a_lower, a_diag, a_upper, d)

            # Rebuild full u_new
            for i in range(1, Nx):
                u_new[i] = U_int[i - 1]

            # Numerical time-step error E_n = max |u_new - u_old|
            max_diff = 0.0
            for i in range(Nx + 1):
                diff = abs(u_new[i] - u_old[i])
                if diff > max_diff:
                    max_diff = diff

            # Write snapshot at time t
            for i in range(Nx + 1):
                f_sol.write(f"{n} {t} {i} {x[i]} {u_new[i]}\n")

            E_list.append(max_diff)
            t_list.append(t)

            # Prepare for next step
            u_old, u_new = u_new, u_old

    # Write convergence data
    with open(output_conv, "w") as f_conv:
        f_conv.write("# General 1D PDE implicit convergence data\n")
        f_conv.write("# col1: n (time index)\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_i |u_i^n - u_i^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("Implicit solution written to:", output_sol)
    print("Convergence history written to:", output_conv)
    print("Example gnuplot commands:")
    print("  plot 'pde1d_implicit_solution.dat' using 4:5 every :::(Nt*(Nx+1)):: \\")
    print("       with linespoints title 'u(x,T)'")
    print("  plot 'pde1d_implicit_convergence.dat' using 2:3 every ::1 \\")
    print("       with linespoints title 'E_n'")

    # Optional matplotlib sanity plots
    if HAVE_MPL:
        # Final-time profile is in u_old after loop
        u_final = u_old[:]

        plt.figure()
        plt.plot(x, u_final, marker='o')
        plt.xlabel("x")
        plt.ylabel("u(x,T)")
        plt.title("General 1D PDE implicit: solution at final time")

        # Convergence plot, skipping n=0 fake zero
        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n = max |u^n - u^{n-1}|")
        plt.title("General 1D PDE implicit: numerical time-step error")
        plt.show()


if __name__ == "__main__":
    main()