"""
2D Heat equation – explicit scheme (FTCS) on a rectangular domain.

PDE:
    u_t = alpha * (u_xx + u_yy),   (x, y) in [0, Lx] × [0, Ly],  t in [0, T]

Grid:
    x_i = i * dx,   i = 0..Nx     where dx = Lx / Nx
    y_j = j * dy,   j = 0..Ny     where dy = Ly / Ny
    t_n = n * dt,   n = 0..Nt     where dt = T / Nt

Finite-difference (explicit FTCS) update for interior points (1 <= i <= Nx-1, 1 <= j <= Ny-1):
    Let
        rx = alpha * dt / dx^2
        ry = alpha * dt / dy^2

    Then
        u_{i,j}^{n+1} = u_{i,j}^n
                        + rx * (u_{i+1,j}^n - 2 u_{i,j}^n + u_{i-1,j}^n)
                        + ry * (u_{i,j+1}^n - 2 u_{i,j}^n + u_{i,j-1}^n)

Boundary conditions: Dirichlet (fixed temperature) on all sides:
    u(0, y, t)    = g_left(y, t)
    u(Lx, y, t)   = g_right(y, t)
    u(x, 0, t)    = g_bottom(x, t)
    u(x, Ly, t)   = g_top(x, t)

Initial condition:
    u(x, y, 0) = u_init(x, y)

Stability guideline (for explicit 2D heat):
    rx + ry <= 1/2   (comment only; not enforced in code, but printed)

----------------------------------------------------------------------
What this script does:
----------------------------------------------------------------------
1. Builds a (Nx+1) × (Ny+1) grid of spatial points.
2. Initializes u[i][j] at t=0 from u_init(x_i, y_j) and then applies
   boundary conditions at t=0.
3. Time-steps using the explicit FTCS scheme.
4. At each time step:
    - Updates interior nodes using the stencil.
    - Applies boundary conditions on all sides at time t_n.
    - Computes a purely numerical "time-step error" measure:
          E_n = max_{i,j} |u_{i,j}^n - u_{i,j}^{n-1}|
5. Writes data to two .dat files:
    (a) 'heat2d_explicit_solution.dat':
        For every time step n and grid node (i, j):
            col1: n        (time index)
            col2: t_n      (current time)
            col3: i        (x index)
            col4: j        (y index)
            col5: x_i
            col6: y_j
            col7: u_{i,j}^n
    (b) 'heat2d_explicit_convergence.dat':
            col1: n        (time index)
            col2: t_n
            col3: E_n = max_{i,j} |u_{i,j}^n - u_{i,j}^{n-1}|

6. Optionally (if matplotlib is available), plots:
    - A 2D color map of u(x, y, T) at final time.
    - A semilog plot of E_n vs t_n (skipping the first zero-error point).

----------------------------------------------------------------------
How to use in the exam:
----------------------------------------------------------------------
- Step 1: Edit the USER PARAMETERS block:
      * Lx, Ly, T, Nx, Ny, Nt, alpha.
- Step 2: Replace u_init(x, y) and boundary functions g_left, g_right,
          g_bottom, g_top with the given problem data.
- Step 3: Run on the lab machine:
        python3 heat2d_explicit.py
- Step 4: Use gnuplot, for example:
      # View final-time temperature distribution (only last time slice n = Nt):
      splot 'heat2d_explicit_solution.dat' using 5:6:7 every :::(Nt*(Nx+1)*(Ny+1)):: with points

      # View convergence vs time (skipping n=0):
      plot 'heat2d_explicit_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'

Note on error/convergence:
- We NEVER use an analytic exact solution.
- Error is purely numerical: E_n = max change in u between consecutive time steps.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
Lx = 1.0      # domain length in x  (0 <= x <= Lx)
Ly = 1.0      # domain length in y  (0 <= y <= Ly)
T  = 0.1      # final time

Nx = 10       # number of intervals in x  -> Nx+1 grid points
Ny = 10       # number of intervals in y  -> Ny+1 grid points
Nt = 100      # number of time steps

alpha = 1.0   # thermal diffusivity

output_sol  = "heat2d_explicit_solution.dat"
output_conv = "heat2d_explicit_convergence.dat"
# ===============================================================


def u_init(x, y):
    """
    Initial condition u(x, y, 0).

    Edit this function in the exam to match the problem.
    Example here: a Gaussian-type bump centered in the domain.
    """
    cx = 0.5 * Lx
    cy = 0.5 * Ly
    return math.exp(-40.0 * ((x - cx) ** 2 + (y - cy) ** 2))


def g_left(y, t):
    """Boundary condition u(0, y, t). Edit in exam if needed."""
    return 0.0


def g_right(y, t):
    """Boundary condition u(Lx, y, t). Edit in exam if needed."""
    return 0.0


def g_bottom(x, t):
    """Boundary condition u(x, 0, t). Edit in exam if needed."""
    return 0.0


def g_top(x, t):
    """Boundary condition u(x, Ly, t). Edit in exam if needed."""
    return 0.0


def main():
    # ------------------------------------------------------------------
    # 1. Build spatial and temporal steps and check stability parameter.
    # ------------------------------------------------------------------
    dx = Lx / float(Nx)
    dy = Ly / float(Ny)
    dt = T / float(Nt)

    rx = alpha * dt / (dx * dx)
    ry = alpha * dt / (dy * dy)

    print(f"2D Heat explicit scheme:")
    print(f"  dx = {dx:.3e}, dy = {dy:.3e}, dt = {dt:.3e}")
    print(f"  rx = alpha * dt / dx^2 = {rx:.3e}")
    print(f"  ry = alpha * dt / dy^2 = {ry:.3e}")
    print("Stability guideline (comment): rx + ry <= 0.5 for explicit scheme.")

    # Build coordinate arrays
    x = [i * dx for i in range(Nx + 1)]
    y = [j * dy for j in range(Ny + 1)]

    # ------------------------------------------------------------------
    # 2. Allocate solution arrays (u_old for t_n, u_new for t_{n+1})
    #    u is stored as u[i][j] with i index in x, j index in y.
    # ------------------------------------------------------------------
    u_old = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]
    u_new = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]

    # ------------------------------------------------------------------
    # 3. Initialize at t = 0 using u_init(x, y) and apply BCs.
    # ------------------------------------------------------------------
    t = 0.0
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            u_old[i][j] = u_init(x[i], y[j])

    # Apply boundary conditions at t=0:
    for j in range(Ny + 1):
        u_old[0][j]   = g_left(y[j], t)
        u_old[Nx][j]  = g_right(y[j], t)
    for i in range(Nx + 1):
        u_old[i][0]   = g_bottom(x[i], t)
        u_old[i][Ny]  = g_top(x[i], t)

    # Lists to store convergence info:
    E_list = []  # time-step error E_n
    t_list = []  # times t_n

    # ------------------------------------------------------------------
    # 4. Open solution file and write initial-time snapshot.
    # ------------------------------------------------------------------
    with open(output_sol, "w") as f_sol:
        f_sol.write("# 2D Heat equation explicit solution (FTCS)\n")
        f_sol.write("# col1: n (time index)\n")
        f_sol.write("# col2: t_n\n")
        f_sol.write("# col3: i (x index)\n")
        f_sol.write("# col4: j (y index)\n")
        f_sol.write("# col5: x_i\n")
        f_sol.write("# col6: y_j\n")
        f_sol.write("# col7: u_ij^n\n")

        # n = 0 snapshot
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                f_sol.write(f"0 {t} {i} {j} {x[i]} {y[j]} {u_old[i][j]}\n")

        E_list.append(0.0)  # defined as 0 at n=0
        t_list.append(t)

        # ------------------------------------------------------------------
        # 5. Time-stepping loop: n = 1..Nt
        # ------------------------------------------------------------------
        for n in range(1, Nt + 1):
            t = n * dt

            # Apply boundary conditions at time t on u_new.
            # We'll fill interior after this.
            for j in range(Ny + 1):
                u_new[0][j]   = g_left(y[j], t)
                u_new[Nx][j]  = g_right(y[j], t)
            for i in range(Nx + 1):
                u_new[i][0]   = g_bottom(x[i], t)
                u_new[i][Ny]  = g_top(x[i], t)

            # Interior update with FTCS stencil:
            max_diff = 0.0
            for i in range(1, Nx):
                for j in range(1, Ny):
                    # Laplacian in x-direction:
                    ux_part = u_old[i + 1][j] - 2.0 * u_old[i][j] + u_old[i - 1][j]
                    # Laplacian in y-direction:
                    uy_part = u_old[i][j + 1] - 2.0 * u_old[i][j] + u_old[i][j - 1]

                    u_new[i][j] = (u_old[i][j]
                                   + rx * ux_part
                                   + ry * uy_part)

                    diff = abs(u_new[i][j] - u_old[i][j])
                    if diff > max_diff:
                        max_diff = diff

            # Write snapshot for time level n
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    f_sol.write(f"{n} {t} {i} {j} {x[i]} {y[j]} {u_new[i][j]}\n")

            # Record numerical time-step error E_n
            E_list.append(max_diff)
            t_list.append(t)

            # Swap old and new for next time step
            u_old, u_new = u_new, u_old

    # ----------------------------------------------------------------------
    # 6. Write convergence data to another .dat file.
    # ----------------------------------------------------------------------
    with open(output_conv, "w") as f_conv:
        f_conv.write("# 2D Heat equation explicit convergence data\n")
        f_conv.write("# col1: n (time index)\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_{i,j} |u_ij^n - u_ij^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("2D Heat explicit solution written to:", output_sol)
    print("Convergence history written to:", output_conv)
    print("Example gnuplot commands:")
    print("  # Final-time 3D scatter:")
    print("  splot 'heat2d_explicit_solution.dat' using 5:6:7 every :::(Nt*(Nx+1)*(Ny+1)):: with points")
    print("  # Convergence vs time (skip n=0):")
    print("  plot 'heat2d_explicit_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'")

    # ----------------------------------------------------------------------
    # 7. Optional matplotlib sanity plots (if available).
    #    We skip the first zero-error point when plotting E_n.
    # ----------------------------------------------------------------------
    if HAVE_MPL:
        # Extract final-time slice from the solution file for plotting.
        # (You could also just use u_old at the end of the time loop,
        #  but here we show how to read from the file if needed.)
        u_final = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]
        with open(output_sol, "r") as f:
            lines = f.readlines()
            # The last (Nx+1)*(Ny+1) data lines correspond to n = Nt
            data_lines = [line for line in lines if not line.startswith("#")]
            last_block = data_lines[-((Nx + 1) * (Ny + 1)):]
            for line in last_block:
                parts = line.split()
                i = int(parts[2])
                j = int(parts[3])
                u_final[i][j] = float(parts[6])

        # Create a simple 2D color map of u(x, y, T)
        X = [[x[i] for j in range(Ny + 1)] for i in range(Nx + 1)]
        Y = [[y[j] for j in range(Ny + 1)] for i in range(Nx + 1)]

        plt.figure()
        # pcolormesh works with nested lists; we pass X, Y, and u_final
        plt.pcolormesh(X, Y, u_final, shading="auto")
        plt.colorbar(label="u(x, y, T)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("2D Heat explicit: temperature at final time")

        # Convergence plot: skip the first zero-error point
        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n = max |u^n - u^{n-1}|")
        plt.title("2D Heat explicit: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()