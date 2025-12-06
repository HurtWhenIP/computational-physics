"""
1D Heat equation – explicit scheme (FTCS).

PDE:
    u_t = alpha * u_xx,   x in [0, L], t in [0, T]

Discretization:
    x_i = i * dx,   i = 0..Nx
    t_n = n * dt,   n = 0..Nt
    r   = alpha * dt / dx^2

Explicit update (FTCS):
    u_i^{n+1} = u_i^n + r * (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)

Boundary conditions:
    u(0, t)   = u_left(t)
    u(L, t)   = u_right(t)

Initial condition:
    u(x, 0) = u_init(x)

What this script does:
- Evolves the solution using FTCS.
- Stability note (in comments only): typically r <= 1/2.
- Tracks a numerical error measure per time step:
      E_n = max_i |u_i^n - u_i^{n-1}|   (n >= 1)
- Writes:
    'heat_explicit_solution.dat' :
        n, t_n, i, x_i, u_i^n
    'heat_explicit_convergence.dat' :
        n, t_n, E_n
- Optionally plots snapshots and error with matplotlib.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * L, T, Nx, Nt, alpha.
- Step 2: Edit u_init, u_left, u_right for the specific problem.
- Step 3: Run:
        python3 heat1d_explicit.py
- Step 4: gnuplot examples:
        # See solution at final time:
        plot 'heat_explicit_solution.dat' using 4:5 every :::(Nt* (Nx+1)):: with linespoints
        # Convergence over time (skip n=0):
        plot 'heat_explicit_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
L = 1.0          # spatial domain [0, L]
T = 0.1          # final time
Nx = 20          # number of spatial intervals
Nt = 100         # number of time steps
alpha = 1.0      # diffusion coefficient

output_sol = "heat_explicit_solution.dat"
output_conv = "heat_explicit_convergence.dat"
# ===============================================================


def u_init(x):
    """Initial condition u(x, 0). Edit for the exam."""
    # Example: Gaussian-like bump
    return math.exp(-50.0 * (x - 0.5) * (x - 0.5))


def u_left(t):
    """Boundary condition at x=0: u(0, t). Edit if needed."""
    return 0.0


def u_right(t):
    """Boundary condition at x=L: u(L, t). Edit if needed."""
    return 0.0


def main():
    dx = L / float(Nx)
    dt = T / float(Nt)
    r = alpha * dt / (dx * dx)

    print(f"Explicit heat scheme: r = {r:.3e} (stability guideline: r <= 0.5)")

    # Spatial grid
    x = [i * dx for i in range(Nx + 1)]

    # Initialize u(x, 0)
    u_old = [u_init(xi) for xi in x]
    u_new = [0.0 for _ in range(Nx + 1)]

    # Apply BC at t=0 (if needed)
    u_old[0] = u_left(0.0)
    u_old[Nx] = u_right(0.0)

    # For storing convergence info
    E_list = []
    t_list = []

    with open(output_sol, "w") as f_sol:
        f_sol.write("# Heat equation explicit solution (FTCS)\n")
        f_sol.write("# col1: n (time index)\n")
        f_sol.write("# col2: t_n\n")
        f_sol.write("# col3: i (space index)\n")
        f_sol.write("# col4: x_i\n")
        f_sol.write("# col5: u_i^n\n")

        # Write initial time slice n=0
        t = 0.0
        for i in range(Nx + 1):
            f_sol.write(f"0 {t} {i} {x[i]} {u_old[i]}\n")

        # Error at n=0 is defined as 0
        E_list.append(0.0)
        t_list.append(t)

        # Time stepping
        for n in range(1, Nt + 1):
            t = n * dt

            # Boundary conditions at time t
            u_new[0] = u_left(t)
            u_new[Nx] = u_right(t)

            # Interior update
            max_diff = 0.0
            for i in range(1, Nx):
                u_new[i] = u_old[i] + r * (u_old[i + 1] - 2.0 * u_old[i] + u_old[i - 1])
                diff = abs(u_new[i] - u_old[i])
                if diff > max_diff:
                    max_diff = diff

            # Write snapshot at time t
            for i in range(Nx + 1):
                f_sol.write(f"{n} {t} {i} {x[i]} {u_new[i]}\n")

            # Record numerical "error" E_n = max_i |u_i^n - u_i^{n-1}|
            E_list.append(max_diff)
            t_list.append(t)

            # Prepare for next step
            u_old, u_new = u_new, u_old

    # Convergence file
    with open(output_conv, "w") as f_conv:
        f_conv.write("# Heat equation FTCS convergence\n")
        f_conv.write("# col1: n (time index)\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_i |u_i^n - u_i^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("Explicit heat solution written to:", output_sol)
    print("Convergence history written to:", output_conv)
    print("Example gnuplot commands:")
    print("  plot 'heat_explicit_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'")

    if HAVE_MPL:
        # Plot final-time profile
        import matplotlib.pyplot as plt
        u_final = []
        with open(output_sol, "r") as f:
            # last (Nx+1) lines correspond to n=Nt
            lines = f.readlines()
            last_lines = lines[-(Nx + 1):]
            for line in last_lines:
                if line.startswith("#"):
                    continue
                parts = line.split()
                u_final.append(float(parts[4]))
        plt.figure()
        plt.plot(x, u_final, marker='o')
        plt.xlabel("x")
        plt.ylabel("u(x, T)")
        plt.title("Heat equation explicit: final profile")

        # Error vs time (skip n=0)
        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n = max |u^n - u^{n-1}|")
        plt.title("Heat explicit: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()