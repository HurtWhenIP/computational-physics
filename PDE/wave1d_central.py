"""
1D Wave equation:

    u_tt = c^2 u_xx,   x in [0, L], t in [0, T]

Finite-difference scheme (central in time, central in space):

Let r = c^2 * dt^2 / dx^2.

For n >= 1, interior nodes i=1..Nx-1:

    u_i^{n+1} = 2 u_i^n - u_i^{n-1}
                + r (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)

Initial data:
    u(x, 0)   = f(x)
    u_t(x, 0) = g(x)

First time step (n=0 -> n=1) is handled by a special formula using g(x).

Boundary conditions (fixed ends):
    u(0, t) = 0,  u(L, t) = 0
    (edit if needed)

What this script does:
- Uses above scheme to compute u_i^n.
- Tracks numerical time-step change:
      E_n = max_i |u_i^n - u_i^{n-1}|
- Writes:
    'wave_solution.dat'      with (n, t_n, i, x_i, u_i^n)
    'wave_convergence.dat'   with (n, t_n, E_n)
- Optionally plots snapshots and error.

How to use:
- Edit L, T, Nx, Nt, c.
- Edit f_init(x), g_init(x).
- Check stability guideline: r <= 1.
- Run:
      python3 wave1d_central.py
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
L = 1.0
T = 1.0
Nx = 50
Nt = 200
c = 1.0

output_sol = "wave_solution.dat"
output_conv = "wave_convergence.dat"
# ===============================================================


def f_init(x):
    """Initial displacement u(x, 0). Edit in the exam."""
    # Example: triangular pulse
    if 0.25 <= x <= 0.75:
        return 1.0 - abs(x - 0.5) / 0.25
    return 0.0


def g_init(x):
    """Initial velocity u_t(x, 0). Edit in the exam."""
    return 0.0


def main():
    dx = L / float(Nx)
    dt = T / float(Nt)
    r = (c * c * dt * dt) / (dx * dx)
    print(f"Wave scheme parameter r = {r:.3e} (stability guideline: r <= 1)")

    x = [i * dx for i in range(Nx + 1)]

    # Allocate three time levels:
    u_prev = [0.0 for _ in range(Nx + 1)]  # u^{n-1}
    u_curr = [0.0 for _ in range(Nx + 1)]  # u^{n}
    u_next = [0.0 for _ in range(Nx + 1)]  # u^{n+1}

    # Initial displacement at t=0: u_curr
    for i in range(Nx + 1):
        u_curr[i] = f_init(x[i])

    # Initial velocity at t=0: g(x)
    # Use a "one-step" formula to get u at t = dt (n=1):
    #   u_i^1 ≈ u_i^0 + dt * g_i + 0.5 * r * (u_{i+1}^0 - 2 u_i^0 + u_{i-1}^0)
    t0 = 0.0
    for i in range(Nx + 1):
        u_prev[i] = u_curr[i]  # store u^0

    # Apply boundary at n=0
    u_curr[0] = 0.0
    u_curr[Nx] = 0.0

    # Compute n=1
    u_next[0] = 0.0
    u_next[Nx] = 0.0
    for i in range(1, Nx):
        g0 = g_init(x[i])
        u_next[i] = (u_curr[i]
                     + dt * g0
                     + 0.5 * r * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]))

    # Now we have:
    #   u_prev = u^0
    #   u_curr = u^0 (but BC-enforced)
    #   u_next = u^1
    # We'll write data as we go and shift time levels.

    E_list = []
    t_list = []

    with open(output_sol, "w") as f_sol:
        f_sol.write("# Wave equation solution\n")
        f_sol.write("# col1: n (time index)\n")
        f_sol.write("# col2: t_n\n")
        f_sol.write("# col3: i (space index)\n")
        f_sol.write("# col4: x_i\n")
        f_sol.write("# col5: u_i^n\n")

        # Write n=0 snapshot
        t = 0.0
        for i in range(Nx + 1):
            f_sol.write(f"0 {t} {i} {x[i]} {u_prev[i]}\n")
        E_list.append(0.0)
        t_list.append(t)

        # Write n=1 snapshot
        t = dt
        # numeric error between n=1 and n=0
        max_diff = 0.0
        for i in range(Nx + 1):
            diff = abs(u_next[i] - u_prev[i])
            if diff > max_diff:
                max_diff = diff
            f_sol.write(f"1 {t} {i} {x[i]} {u_next[i]}\n")
        E_list.append(max_diff)
        t_list.append(t)

        # Shift levels:
        # After this, we want:
        #   u_prev = u^0
        #   u_curr = u^1
        u_prev = u_prev
        u_curr = u_next[:]

        # Time stepping from n=1 to Nt-1 to compute up to n=Nt
        for n in range(2, Nt + 1):
            t = n * dt

            # Zero boundary
            u_next[0] = 0.0
            u_next[Nx] = 0.0

            # Interior update:
            for i in range(1, Nx):
                u_next[i] = (2.0 * u_curr[i] - u_prev[i]
                             + r * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]))

            # Numerical "error" vs previous time level
            max_diff = 0.0
            for i in range(Nx + 1):
                diff = abs(u_next[i] - u_curr[i])
                if diff > max_diff:
                    max_diff = diff

            # Write snapshot
            for i in range(Nx + 1):
                f_sol.write(f"{n} {t} {i} {x[i]} {u_next[i]}\n")

            E_list.append(max_diff)
            t_list.append(t)

            # Shift: u_prev <- u_curr, u_curr <- u_next
            u_prev, u_curr, u_next = u_curr, u_next, u_prev

    with open(output_conv, "w") as f_conv:
        f_conv.write("# Wave equation convergence\n")
        f_conv.write("# col1: n\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_i |u^n - u^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("Wave equation solution written to:", output_sol)
    print("Convergence written to:", output_conv)
    print("Example gnuplot:")
    print("  plot 'wave_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'")

    if HAVE_MPL:
        import matplotlib.pyplot as plt

        # final profile
        u_final = []
        with open(output_sol, "r") as f:
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
        plt.title("Wave equation: final displacement")

        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n")
        plt.title("Wave equation: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()