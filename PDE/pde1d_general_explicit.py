"""
General 1D time-dependent PDE – explicit finite difference template.

PDE form (1D, time-dependent):
    u_t = a(x, t, u) * u_xx
          + b(x, t, u) * u_x
          + c(x, t, u) * u
          + s(x, t, u)

Domain:
    x in [0, L],  t in [0, T]

Discretization:
    x_i = i * dx,   i = 0..Nx          where dx = L / Nx
    t_n = n * dt,   n = 0..Nt          where dt = T / Nt

Finite differences:
    u_i^n      ≈ u(x_i, t_n)
    u_xx^n,i   ≈ (u_{i+1}^n - 2 u_i^n + u_{i-1}^n) / dx^2
    u_x^n,i    ≈ (u_{i+1}^n - u_{i-1}^n) / (2 dx)

Explicit update for interior points (i = 1..Nx-1):
    u_i^{n+1} = u_i^n + dt * [
                    a_i * u_xx^n,i
                  + b_i * u_x^n,i
                  + c_i * u_i^n
                  + s_i
               ]

where:
    a_i = a(x_i, t_n, u_i^n), etc.

Boundary conditions:
    Here we implement Dirichlet BC:
        u(0, t)   = g_left(t)
        u(L, t)   = g_right(t)
    You can modify the update near boundaries for Neumann/Robin if needed.

----------------------------------------------------------------------
What this script does:
----------------------------------------------------------------------
1. Defines a general PDE structure via coefficient functions:
       a(x,t,u), b(x,t,u), c(x,t,u), s(x,t,u)
2. Uses explicit finite-difference time stepping for u_t.
3. Uses Dirichlet boundary conditions given by g_left(t), g_right(t).
4. Tracks purely numerical convergence:
       E_n = max_i |u_i^n - u_i^{n-1}|
   (NO analytic exact solution used.)
5. Writes two .dat files:
   (a) 'pde1d_solution.dat':
       - col1: n      (time index)
       - col2: t_n    (time)
       - col3: i      (space index)
       - col4: x_i
       - col5: u_i^n
   (b) 'pde1d_convergence.dat':
       - col1: n
       - col2: t_n
       - col3: E_n = max_i |u_i^n - u_i^{n-1}|

6. Optionally (if matplotlib is available) plots:
   - u(x, T) vs x at final time.
   - Semilog plot of E_n vs t_n (skipping the n=0 point so you don't get a fake 0).

----------------------------------------------------------------------
How to use in the exam:
----------------------------------------------------------------------
- Step 1: In USER PARAMETERS below, set:
      L, T, Nx, Nt, and choose an output filename.
- Step 2: Replace the coefficient functions a, b, c, s and u_init(x)
          and boundary functions g_left(t), g_right(t) to match your PDE.
- Step 3: Run:
      python3 pde1d_general_explicit.py
- Step 4: Gnuplot examples:

    # Final time profile (only last time slice):
    # (Nt+1)*(Nx+1) total lines; last block is the final time.
    plot 'pde1d_solution.dat' using 4:5 every :::(Nt*(Nx+1)):: with linespoints title 'u(x,T)'

    # Convergence vs time (skip the first n=0 row):
    plot 'pde1d_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'

Note on stability:
- For diffusion-dominated problems (like heat):
      dt * max(a) / dx^2  <= 1/2   (rough guideline)
- For advection-dominated problems:
      dt * max(|b|) / dx  <= 1     (CFL-like condition)
These are NOT enforced by the code; you must choose dt, dx accordingly.
"""

import math

# Optional plotting for sanity checks (ignored if matplotlib is not installed)
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

output_sol  = "pde1d_solution.dat"
output_conv = "pde1d_convergence.dat"
# ===============================================================


# ---------- PDE COEFFICIENTS & SOURCE: EDIT FOR YOUR PROBLEM ----------

def a_coeff(x, t, u):
    """
    Diffusion-like coefficient a(x,t,u) for the u_xx term.

    Examples:
      - Heat equation:       a = alpha (constant)
      - Advection-only:      a = 0
      - Reaction-diffusion:  a = D (constant or function)

    For a simple heat equation with alpha = 1:
        return 1.0
    """
    return 1.0   # EDIT in exam if needed


def b_coeff(x, t, u):
    """
    Advection-like coefficient b(x,t,u) for the u_x term.

    Examples:
      - Pure diffusion: b = 0
      - Constant advection speed v: b = v
    """
    return 0.0   # EDIT in exam if needed


def c_coeff(x, t, u):
    """
    Reaction-like coefficient c(x,t,u) multiplying u.

    Examples:
      - No reaction:    c = 0
      - Decay term:     c = -lambda
    """
    return 0.0   # EDIT in exam if needed


def s_source(x, t, u):
    """
    Source term s(x,t,u). For no source, return 0.0.
    """
    return 0.0   # EDIT in exam if needed


# ---------- INITIAL & BOUNDARY CONDITIONS: EDIT FOR YOUR PROBLEM ----------

def u_init(x):
    """
    Initial condition u(x, 0).

    Example here: a bump in the center.
    Replace with the given initial condition in the exam.
    """
    return math.exp(-50.0 * (x - 0.5 * L) ** 2)


def g_left(t):
    """
    Dirichlet boundary at x = 0: u(0, t) = g_left(t).
    """
    return 0.0   # EDIT if needed


def g_right(t):
    """
    Dirichlet boundary at x = L: u(L, t) = g_right(t).
    """
    return 0.0   # EDIT if needed


# ----------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------

def main():
    # Step sizes
    dx = L / float(Nx)
    dt = T / float(Nt)

    print("General 1D PDE explicit solver")
    print(f"  dx = {dx:.3e}, dt = {dt:.3e}")
    print("  PDE: u_t = a u_xx + b u_x + c u + s")
    print("  NOTE: Stability depends on your a, b, c, s and chosen dt, dx.")

    # Spatial grid
    x = [i * dx for i in range(Nx + 1)]

    # Allocate solution arrays
    u_old = [0.0 for _ in range(Nx + 1)]
    u_new = [0.0 for _ in range(Nx + 1)]

    # Initial condition at t=0
    t = 0.0
    for i in range(Nx + 1):
        u_old[i] = u_init(x[i])

    # Apply boundary conditions at t=0
    u_old[0]   = g_left(t)
    u_old[Nx]  = g_right(t)

    # Lists for convergence data
    E_list = []    # E_n = max |u^n - u^{n-1}|
    t_list = []    # time levels

    # Open solution file and write initial profile
    with open(output_sol, "w") as f_sol:
        f_sol.write("# General 1D PDE explicit solution\n")
        f_sol.write("# col1: n (time index)\n")
        f_sol.write("# col2: t_n\n")
        f_sol.write("# col3: i (space index)\n")
        f_sol.write("# col4: x_i\n")
        f_sol.write("# col5: u_i^n\n")

        # n = 0 snapshot
        for i in range(Nx + 1):
            f_sol.write(f"0 {t} {i} {x[i]} {u_old[i]}\n")

        E_list.append(0.0)   # defined as 0 at n=0 (no previous step)
        t_list.append(t)

        # Time-stepping loop
        for n in range(1, Nt + 1):
            t = n * dt

            # Apply boundary conditions at current time on u_new
            u_new[0]  = g_left(t)
            u_new[Nx] = g_right(t)

            # Interior points i = 1..Nx-1
            max_diff = 0.0

            for i in range(1, Nx):
                ui   = u_old[i]
                uim1 = u_old[i - 1]
                uip1 = u_old[i + 1]

                # Second derivative u_xx
                u_xx = (uip1 - 2.0 * ui + uim1) / (dx * dx)

                # First derivative u_x (central)
                u_x = (uip1 - uim1) / (2.0 * dx)

                # Evaluate coefficients at (x_i, t_{n-1}, u_i^n)
                a_val = a_coeff(x[i], t - dt, ui)
                b_val = b_coeff(x[i], t - dt, ui)
                c_val = c_coeff(x[i], t - dt, ui)
                s_val = s_source(x[i], t - dt, ui)

                # Explicit Euler in time
                u_new[i] = ui + dt * (a_val * u_xx +
                                      b_val * u_x +
                                      c_val * ui +
                                      s_val)

                # Track max difference for numerical "error" between steps
                diff = abs(u_new[i] - u_old[i])
                if diff > max_diff:
                    max_diff = diff

            # Write snapshot at time t
            for i in range(Nx + 1):
                f_sol.write(f"{n} {t} {i} {x[i]} {u_new[i]}\n")

            # Store numerical "error" for this step
            E_list.append(max_diff)
            t_list.append(t)

            # Prepare for next time step
            u_old, u_new = u_new, u_old

    # Write convergence data to separate file
    with open(output_conv, "w") as f_conv:
        f_conv.write("# General 1D PDE explicit convergence data\n")
        f_conv.write("# col1: n (time index)\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_i |u_i^n - u_i^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("Solution written to:", output_sol)
    print("Convergence history written to:", output_conv)
    print("Example gnuplot commands:")
    print("  # Final time profile (last block only):")
    print("  plot 'pde1d_solution.dat' using 4:5 every :::(Nt*(Nx+1)):: with linespoints title 'u(x,T)'")
    print("  # Convergence vs time (skip n=0):")
    print("  plot 'pde1d_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'")

    # Optional matplotlib sanity plots
    if HAVE_MPL:
        # Final-time profile is just u_old after loop
        # (because we swapped u_old, u_new at the end)
        u_final = u_old[:]

        plt.figure()
        plt.plot(x, u_final, marker='o')
        plt.xlabel("x")
        plt.ylabel("u(x, T)")
        plt.title("General 1D PDE: solution at final time")

        # Convergence plot: skip the first point (n=0, E_0 = 0)
        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n = max |u^n - u^{n-1}|")
        plt.title("General 1D PDE: numerical time-step error")
        plt.show()


if __name__ == "__main__":
    main()