"""
2D Heat equation – implicit scheme (Backward Euler) on a rectangular domain.

PDE:
    u_t = alpha * (u_xx + u_yy),   (x, y) in [0, Lx] × [0, Ly],  t in [0, T]

Grid:
    x_i = i * dx,   i = 0..Nx
    y_j = j * dy,   j = 0..Ny
    t_n = n * dt,   n = 0..Nt

Implicit Backward Euler scheme:
    (u_ij^{n+1} - u_ij^n) / dt = alpha * (u_xx^{n+1} + u_yy^{n+1})

Discretizing Laplacian at time level n+1:
    u_xx^{n+1} ≈ (u_{i+1,j}^{n+1} - 2 u_{i,j}^{n+1} + u_{i-1,j}^{n+1}) / dx^2
    u_yy^{n+1} ≈ (u_{i,j+1}^{n+1} - 2 u_{i,j}^{n+1} + u_{i,j-1}^{n+1}) / dy^2

Let:
    r_x = alpha * dt / dx^2
    r_y = alpha * dt / dy^2

Then for interior points (1 <= i <= Nx-1, 1 <= j <= Ny-1):

    - r_x u_{i-1,j}^{n+1} - r_y u_{i,j-1}^{n+1}
    + (1 + 2 r_x + 2 r_y) u_{i,j}^{n+1}
    - r_x u_{i+1,j}^{n+1} - r_y u_{i,j+1}^{n+1}
    = u_{i,j}^n

This gives, at each time step, a large linear system:
    A * U^{n+1}_int = d

where:
    - U^{n+1}_int is the vector of interior unknowns (flattened (i,j)).
    - A is time-independent (same for every step).
    - d depends on u^n and boundary values at time t_{n+1}.

Boundary conditions (Dirichlet, fixed temperature):
    u(0,  y, t) = g_left(y, t)
    u(Lx, y, t) = g_right(y, t)
    u(x,  0, t) = g_bottom(x, t)
    u(x, Ly, t) = g_top(x, t)

Initial condition:
    u(x, y, 0) = u_init(x, y)

----------------------------------------------------------------------
What this script does:
----------------------------------------------------------------------
1. Builds uniform grid in space and time.
2. Builds matrix A (size M x M, with M = (Nx-1)*(Ny-1)) representing the
   5-point stencil for all interior nodes.
3. At each time step:
    - Builds RHS vector d from previous time layer u^n and boundary data.
    - Solves A U^{n+1}_int = d using basic Gaussian elimination.
    - Reconstructs full u^{n+1}(i,j), including boundaries.
    - Computes purely numerical time-step error:
          E_n = max_{i,j} |u_ij^{n+1} - u_ij^n|
4. Writes:
    'heat2d_implicit_solution.dat'
        col1: n   (time index)
        col2: t_n
        col3: i   (x index)
        col4: j   (y index)
        col5: x_i
        col6: y_j
        col7: u_ij^n

    'heat2d_implicit_convergence.dat'
        col1: n
        col2: t_n
        col3: E_n

5. Optionally (if matplotlib is available) plots:
    - Color map of u(x, y, T) at final time.
    - Semilog plot of E_n vs t_n (skipping n=0).

----------------------------------------------------------------------
How to use in the exam:
----------------------------------------------------------------------
- Step 1: Edit USER PARAMETERS:
      Lx, Ly, T, Nx, Ny, Nt, alpha.
- Step 2: Change u_init, g_left, g_right, g_bottom, g_top as per problem.
- Step 3: Run:
        python3 heat2d_implicit.py
- Step 4: Example gnuplot commands:
        splot 'heat2d_implicit_solution.dat' using 5:6:7 every :::(Nt*(Nx+1)*(Ny+1)):: with points
        plot  'heat2d_implicit_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'

Note: This template uses a dense matrix + Gaussian elimination. For exam-sized
Nx, Ny (like 5, 10, maybe 15), this is fine.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
Lx = 1.0
Ly = 1.0
T  = 0.1

Nx = 5           # number of intervals in x  -> Nx+1 points
Ny = 5           # number of intervals in y  -> Ny+1 points
Nt = 20          # number of time steps

alpha = 1.0

output_sol  = "heat2d_implicit_solution.dat"
output_conv = "heat2d_implicit_convergence.dat"

PRINT_MATRIX = False   # set True if you want to print A once
# ===============================================================


def u_init(x, y):
    """Initial condition u(x, y, 0). Edit for the exam."""
    # Example: simple peak at center
    cx = 0.5 * Lx
    cy = 0.5 * Ly
    return math.exp(-20.0 * ((x - cx) ** 2 + (y - cy) ** 2))


def g_left(y, t):
    """Boundary at x = 0."""
    return 0.0


def g_right(y, t):
    """Boundary at x = Lx."""
    return 0.0


def g_bottom(x, t):
    """Boundary at y = 0."""
    return 0.0


def g_top(x, t):
    """Boundary at y = Ly."""
    return 0.0


def ij_to_k(i, j, Ny):
    """
    Map interior node indices (i, j) with 1 <= i <= Nx-1, 1 <= j <= Ny-1
    to a single index k = 0..M-1, where M = (Nx-1)*(Ny-1).

    Here we use row-major ordering:
        k = (i-1)*(Ny-1) + (j-1)
    """
    return (i - 1) * (Ny - 1) + (j - 1)


def build_matrix_A(rx, ry, Nx, Ny):
    """
    Build the coefficient matrix A (size M x M, M = (Nx-1)*(Ny-1))
    for the 2D implicit heat scheme:

    For each interior node (i, j):
        - r_x u_{i-1,j}^{n+1} - r_y u_{i,j-1}^{n+1}
        + (1 + 2 r_x + 2 r_y) u_{i,j}^{n+1}
        - r_x u_{i+1,j}^{n+1} - r_y u_{i,j+1}^{n+1} = u_{i,j}^n

    So:
        main diagonal: (1 + 2 r_x + 2 r_y)
        neighbors in x-direction: -r_x
        neighbors in y-direction: -r_y
    """
    M = (Nx - 1) * (Ny - 1)
    A = [[0.0 for _ in range(M)] for _ in range(M)]

    for i in range(1, Nx):
        for j in range(1, Ny):
            k = ij_to_k(i, j, Ny)

            # main diagonal
            A[k][k] = 1.0 + 2.0 * rx + 2.0 * ry

            # neighbor in +x direction -> (i+1, j)
            if i + 1 <= Nx - 1:
                k_right = ij_to_k(i + 1, j, Ny)
                A[k][k_right] = -rx

            # neighbor in -x direction -> (i-1, j)
            if i - 1 >= 1:
                k_left = ij_to_k(i - 1, j, Ny)
                A[k][k_left] = -rx

            # neighbor in +y direction -> (i, j+1)
            if j + 1 <= Ny - 1:
                k_up = ij_to_k(i, j + 1, Ny)
                A[k][k_up] = -ry

            # neighbor in -y direction -> (i, j-1)
            if j - 1 >= 1:
                k_down = ij_to_k(i, j - 1, Ny)
                A[k][k_down] = -ry

    if PRINT_MATRIX:
        print("2D implicit heat matrix A:")
        for row in A:
            print("  ", row)

    return A


def gaussian_elimination_solve(A, d):
    """
    Basic Gaussian elimination (no pivoting) to solve A x = d.
    Matrix A and RHS d are modified in-place.
    """
    n = len(A)

    # Forward elimination
    for k in range(n - 1):
        if abs(A[k][k]) < 1e-14:
            print("Warning: near-zero pivot encountered in 2D implicit solver.")
        m = A[k][k]
        if abs(m) < 1e-14:
            continue
        for i in range(k + 1, n):
            factor = A[i][k] / m
            A[i][k] = 0.0
            for j in range(k + 1, n):
                A[i][j] -= factor * A[k][j]
            d[i] -= factor * d[k]

    # Back substitution
    x = [0.0 for _ in range(n)]
    if abs(A[n - 1][n - 1]) < 1e-14:
        print("Warning: near-zero diagonal at last row.")
        x[n - 1] = 0.0
    else:
        x[n - 1] = d[n - 1] / A[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * x[j]
        if abs(A[i][i]) < 1e-14:
            print("Warning: near-zero diagonal in back substitution.")
            x[i] = 0.0
        else:
            x[i] = (d[i] - s) / A[i][i]

    return x


def main():
    # Step sizes
    dx = Lx / float(Nx)
    dy = Ly / float(Ny)
    dt = T / float(Nt)

    rx = alpha * dt / (dx * dx)
    ry = alpha * dt / (dy * dy)

    print("2D Heat implicit (Backward Euler)")
    print(f"  dx = {dx:.3e}, dy = {dy:.3e}, dt = {dt:.3e}")
    print(f"  r_x = {rx:.3e}, r_y = {ry:.3e}")
    print("Implicit scheme is unconditionally stable in theory (no CFL limit).")

    # Grids
    x = [i * dx for i in range(Nx + 1)]
    y = [j * dy for j in range(Ny + 1)]

    # Allocate solution arrays (full grid)
    u_old = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]
    u_new = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]

    # Initial condition at t=0
    t = 0.0
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            u_old[i][j] = u_init(x[i], y[j])

    # Apply Dirichlet boundary conditions at t=0
    for j in range(Ny + 1):
        u_old[0][j]  = g_left(y[j], t)
        u_old[Nx][j] = g_right(y[j], t)
    for i in range(Nx + 1):
        u_old[i][0]  = g_bottom(x[i], t)
        u_old[i][Ny] = g_top(x[i], t)

    # Build matrix A once
    A_template = build_matrix_A(rx, ry, Nx, Ny)
    M = (Nx - 1) * (Ny - 1)

    # Convergence storage
    E_list = []
    t_list = []

    # Open solution file
    with open(output_sol, "w") as f_sol:
        f_sol.write("# 2D Heat equation implicit (Backward Euler) solution\n")
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

        E_list.append(0.0)
        t_list.append(t)

        # Time stepping
        for n in range(1, Nt + 1):
            t = n * dt

            # Build RHS vector d from u_old and boundary values at time t
            d = [0.0 for _ in range(M)]

            # Apply boundary conditions at time t in u_new for now
            for j in range(Ny + 1):
                u_new[0][j]  = g_left(y[j], t)
                u_new[Nx][j] = g_right(y[j], t)
            for i in range(Nx + 1):
                u_new[i][0]  = g_bottom(x[i], t)
                u_new[i][Ny] = g_top(x[i], t)

            # Fill d for interior nodes:
            for i in range(1, Nx):
                for j in range(1, Ny):
                    k = ij_to_k(i, j, Ny)
                    d[k] = u_old[i][j]

                    # boundary contributions: neighbors that are boundaries
                    # (i-1, j) is boundary if i-1=0
                    if i - 1 == 0:
                        d[k] += rx * u_new[0][j]
                    # (i+1, j) is boundary if i+1=Nx
                    if i + 1 == Nx:
                        d[k] += rx * u_new[Nx][j]
                    # (i, j-1) boundary if j-1=0
                    if j - 1 == 0:
                        d[k] += ry * u_new[i][0]
                    # (i, j+1) boundary if j+1=Ny
                    if j + 1 == Ny:
                        d[k] += ry * u_new[i][Ny]

            # Solve A * u_int_new = d
            A = [row[:] for row in A_template]
            u_int_new = gaussian_elimination_solve(A, d)

            # Rebuild full u_new from interior vector
            for i in range(1, Nx):
                for j in range(1, Ny):
                    k = ij_to_k(i, j, Ny)
                    u_new[i][j] = u_int_new[k]

            # Compute numerical time-step error E_n = max |u_new - u_old|
            max_diff = 0.0
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    diff = abs(u_new[i][j] - u_old[i][j])
                    if diff > max_diff:
                        max_diff = diff

            # Write snapshot at time t
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    f_sol.write(f"{n} {t} {i} {j} {x[i]} {y[j]} {u_new[i][j]}\n")

            E_list.append(max_diff)
            t_list.append(t)

            # Prepare for next step
            u_old, u_new = u_new, u_old

    # Convergence file
    with open(output_conv, "w") as f_conv:
        f_conv.write("# 2D Heat implicit (Backward Euler) convergence data\n")
        f_conv.write("# col1: n\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_{i,j} |u_ij^n - u_ij^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("2D implicit heat solution written to:", output_sol)
    print("Convergence history written to:", output_conv)
    print("Example gnuplot:")
    print("  splot 'heat2d_implicit_solution.dat' using 5:6:7 every :::(Nt*(Nx+1)*(Ny+1)):: with points")
    print("  plot  'heat2d_implicit_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'")

    # Optional matplotlib sanity plots
    if HAVE_MPL:
        # Re-read final-time slice
        u_final = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]
        with open(output_sol, "r") as f:
            lines = [ln for ln in f.readlines() if not ln.startswith("#")]
            # last (Nx+1)*(Ny+1) lines correspond to n = Nt
            block = lines[-((Nx + 1) * (Ny + 1)):]
            for line in block:
                parts = line.split()
                i = int(parts[2])
                j = int(parts[3])
                u_final[i][j] = float(parts[6])

        X = [[x[i] for j in range(Ny + 1)] for i in range(Nx + 1)]
        Y = [[y[j] for j in range(Ny + 1)] for i in range(Nx + 1)]

        plt.figure()
        plt.pcolormesh(X, Y, u_final, shading="auto")
        plt.colorbar(label="u(x, y, T)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("2D Heat implicit: final temperature")

        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n")
        plt.title("2D Heat implicit: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()