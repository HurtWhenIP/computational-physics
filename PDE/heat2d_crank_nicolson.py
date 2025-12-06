"""
2D Heat equation – Crank–Nicolson scheme on a rectangular domain.

PDE:
    u_t = alpha * (u_xx + u_yy),   (x, y) in [0, Lx] × [0, Ly],  t in [0, T]

Grid:
    x_i = i * dx,   i = 0..Nx
    y_j = j * dy,   j = 0..Ny
    t_n = n * dt,   n = 0..Nt

Crank–Nicolson in 2D (time-centered, space-centered):

We combine explicit and implicit 2D Laplacians:

Let
    r_x = alpha * dt / dx^2
    r_y = alpha * dt / dy^2

Interior equation (1 <= i <= Nx-1, 1 <= j <= Ny-1):

Left-hand side (implicit at n+1):
    - (r_x/2) u_{i-1,j}^{n+1} - (r_y/2) u_{i,j-1}^{n+1}
    + (1 + r_x + r_y) u_{i,j}^{n+1}
    - (r_x/2) u_{i+1,j}^{n+1} - (r_y/2) u_{i,j+1}^{n+1}

Right-hand side (explicit at n):
    + (r_x/2) u_{i-1,j}^n + (r_y/2) u_{i,j-1}^n
    + (1 - r_x - r_y) u_{i,j}^n
    + (r_x/2) u_{i+1,j}^n + (r_y/2) u_{i,j+1}^n

That can be written in matrix form:
    A * U^{n+1}_int = B * U^n_int + boundary_contributions

where:
    - U_int is the vector of interior unknowns (flattened (i, j)).
    - A is a 5-point stencil matrix with main diag (1 + r_x + r_y)
      and off-diags -r_x/2, -r_y/2.
    - B is another 5-point stencil with main diag (1 - r_x - r_y)
      and off-diags +r_x/2, +r_y/2.

Boundary conditions (Dirichlet):
    u(0,  y, t) = g_left(y, t)
    u(Lx, y, t) = g_right(y, t)
    u(x,  0, t) = g_bottom(x, t)
    u(x, Ly, t) = g_top(x, t)

Initial condition:
    u(x, y, 0) = u_init(x, y)

----------------------------------------------------------------------
What this script does:
----------------------------------------------------------------------
1. Builds uniform space-time grids.
2. Builds two matrices A and B (size M x M, M = (Nx-1)*(Ny-1)).
3. At each time step:
    - Forms interior vector U^n_int from u^n.
    - Computes RHS: d = B * U^n_int + boundary terms (both at n and n+1).
    - Solves A U^{n+1}_int = d via Gaussian elimination.
    - Reassembles full grid u^{n+1}(i,j).
    - Computes numerical error:
          E_n = max_{i,j} |u_ij^{n+1} - u_ij^n|.
4. Writes:
    'heat2d_cn_solution.dat'
        col1: n, col2: t_n, col3: i, col4: j, col5: x_i, col6: y_j, col7: u_ij^n
    'heat2d_cn_convergence.dat'
        col1: n, col2: t_n, col3: E_n

5. Optional matplotlib:
    - Color map of u(x, y, T)
    - Semilog plot of E_n vs t_n (skip n=0 in the plot)

----------------------------------------------------------------------
How to use in the exam:
----------------------------------------------------------------------
- Edit USER PARAMETERS (Lx, Ly, T, Nx, Ny, Nt, alpha).
- Edit u_init, g_left, g_right, g_bottom, g_top.
- Run:
      python3 heat2d_crank_nicolson.py
- Example gnuplot:
      splot 'heat2d_cn_solution.dat' using 5:6:7 every :::(Nt*(Nx+1)*(Ny+1)):: with points
      plot  'heat2d_cn_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'
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

Nx = 5       # intervals in x
Ny = 5       # intervals in y
Nt = 20      # time steps

alpha = 1.0

output_sol  = "heat2d_cn_solution.dat"
output_conv = "heat2d_cn_convergence.dat"

PRINT_MATRICES = False
# ===============================================================


def u_init(x, y):
    """Initial condition u(x, y, 0). Edit for the exam."""
    return math.sin(math.pi * x) * math.sin(math.pi * y)


def g_left(y, t):
    return 0.0


def g_right(y, t):
    return 0.0


def g_bottom(x, t):
    return 0.0


def g_top(x, t):
    return 0.0


def ij_to_k(i, j, Ny):
    """
    Map interior node (i, j), 1 <= i <= Nx-1, 1 <= j <= Ny-1,
    to linear index k: 0..M-1, M=(Nx-1)*(Ny-1).
    """
    return (i - 1) * (Ny - 1) + (j - 1)


def build_matrices_A_B(rx, ry, Nx, Ny):
    """
    Build matrices A and B for 2D Crank–Nicolson.

    A corresponds to coefficients at time n+1:
        main diag: 1 + r_x + r_y
        neighbors in x: -r_x/2
        neighbors in y: -r_y/2

    B corresponds to coefficients at time n:
        main diag: 1 - r_x - r_y
        neighbors in x: +r_x/2
        neighbors in y: +r_y/2
    """
    M = (Nx - 1) * (Ny - 1)
    A = [[0.0 for _ in range(M)] for _ in range(M)]
    B = [[0.0 for _ in range(M)] for _ in range(M)]

    for i in range(1, Nx):
        for j in range(1, Ny):
            k = ij_to_k(i, j, Ny)

            # main diagonal
            A[k][k] = 1.0 + rx + ry
            B[k][k] = 1.0 - rx - ry

            # +x neighbor
            if i + 1 <= Nx - 1:
                k_right = ij_to_k(i + 1, j, Ny)
                A[k][k_right] = -0.5 * rx
                B[k][k_right] =  0.5 * rx

            # -x neighbor
            if i - 1 >= 1:
                k_left = ij_to_k(i - 1, j, Ny)
                A[k][k_left] = -0.5 * rx
                B[k][k_left] =  0.5 * rx

            # +y neighbor
            if j + 1 <= Ny - 1:
                k_up = ij_to_k(i, j + 1, Ny)
                A[k][k_up] = -0.5 * ry
                B[k][k_up] =  0.5 * ry

            # -y neighbor
            if j - 1 >= 1:
                k_down = ij_to_k(i, j - 1, Ny)
                A[k][k_down] = -0.5 * ry
                B[k][k_down] =  0.5 * ry

    if PRINT_MATRICES:
        print("Matrix A (Crank–Nicolson 2D):")
        for row in A:
            print("  ", row)
        print("Matrix B (Crank–Nicolson 2D):")
        for row in B:
            print("  ", row)

    return A, B


def mat_vec_mult(M, v):
    n = len(v)
    res = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += M[i][j] * v[j]
        res[i] = s
    return res


def gaussian_elimination_solve(A, d):
    """
    Basic dense Gaussian elimination for A x = d (no pivoting).
    OK for small M in exam problems.
    """
    n = len(A)

    # Forward elimination
    for k in range(n - 1):
        if abs(A[k][k]) < 1e-14:
            print("Warning: near-zero pivot in 2D CN elimination.")
        pivot = A[k][k]
        if abs(pivot) < 1e-14:
            continue
        for i in range(k + 1, n):
            factor = A[i][k] / pivot
            A[i][k] = 0.0
            for j in range(k + 1, n):
                A[i][j] -= factor * A[k][j]
            d[i] -= factor * d[k]

    # Back substitution
    x = [0.0] * n
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
    dx = Lx / float(Nx)
    dy = Ly / float(Ny)
    dt = T / float(Nt)

    rx = alpha * dt / (dx * dx)
    ry = alpha * dt / (dy * dy)

    print("2D Heat Crank–Nicolson")
    print(f"  dx = {dx:.3e}, dy = {dy:.3e}, dt = {dt:.3e}")
    print(f"  r_x = {rx:.3e}, r_y = {ry:.3e}")

    x = [i * dx for i in range(Nx + 1)]
    y = [j * dy for j in range(Ny + 1)]

    # Full-grid solutions
    u_old = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]
    u_new = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]

    # Initial condition at t=0
    t = 0.0
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            u_old[i][j] = u_init(x[i], y[j])

    # Apply BC at t=0
    for j in range(Ny + 1):
        u_old[0][j]  = g_left(y[j], t)
        u_old[Nx][j] = g_right(y[j], t)
    for i in range(Nx + 1):
        u_old[i][0]  = g_bottom(x[i], t)
        u_old[i][Ny] = g_top(x[i], t)

    # Build matrices A and B
    A_template, B = build_matrices_A_B(rx, ry, Nx, Ny)
    M = (Nx - 1) * (Ny - 1)

    E_list = []
    t_list = []

    with open(output_sol, "w") as f_sol:
        f_sol.write("# 2D Heat equation Crank–Nicolson solution\n")
        f_sol.write("# col1: n\n")
        f_sol.write("# col2: t_n\n")
        f_sol.write("# col3: i\n")
        f_sol.write("# col4: j\n")
        f_sol.write("# col5: x_i\n")
        f_sol.write("# col6: y_j\n")
        f_sol.write("# col7: u_ij^n\n")

        # n=0 snapshot
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                f_sol.write(f"0 {t} {i} {j} {x[i]} {y[j]} {u_old[i][j]}\n")
        E_list.append(0.0)
        t_list.append(t)

        # Time stepping
        for n in range(1, Nt + 1):
            t = n * dt

            # Boundary values at times n and n-1
            # (Dirichlet may be time-dependent; we recompute both.)
            # u_old corresponds to time t - dt
            t_old = t - dt
            # apply old BC to u_old explicitly (in case they are time-dependent)
            for j in range(Ny + 1):
                u_old[0][j]  = g_left(y[j], t_old)
                u_old[Nx][j] = g_right(y[j], t_old)
            for i in range(Nx + 1):
                u_old[i][0]  = g_bottom(x[i], t_old)
                u_old[i][Ny] = g_top(x[i], t_old)

            # apply new BC into u_new (only boundaries for now)
            for j in range(Ny + 1):
                u_new[0][j]  = g_left(y[j], t)
                u_new[Nx][j] = g_right(y[j], t)
            for i in range(Nx + 1):
                u_new[i][0]  = g_bottom(x[i], t)
                u_new[i][Ny] = g_top(x[i], t)

            # Build U_old_int vector
            U_old_int = [0.0 for _ in range(M)]
            for i in range(1, Nx):
                for j in range(1, Ny):
                    k = ij_to_k(i, j, Ny)
                    U_old_int[k] = u_old[i][j]

            # Compute B * U_old_int
            BU = mat_vec_mult(B, U_old_int)

            # Build RHS d = B * U_old_int + boundary contributions
            d = BU[:]

            # boundary contribs (from both n and n+1 footprints)
            for i in range(1, Nx):
                for j in range(1, Ny):
                    k = ij_to_k(i, j, Ny)

                    # neighbor (i-1, j)
                    if i - 1 == 0:
                        # appears with + (r_x/2) at time n and + (r_x/2) at time n+1 (moved to RHS)
                        d[k] += 0.5 * rx * (u_old[0][j] + u_new[0][j])
                    # neighbor (i+1, j)
                    if i + 1 == Nx:
                        d[k] += 0.5 * rx * (u_old[Nx][j] + u_new[Nx][j])
                    # neighbor (i, j-1)
                    if j - 1 == 0:
                        d[k] += 0.5 * ry * (u_old[i][0] + u_new[i][0])
                    # neighbor (i, j+1)
                    if j + 1 == Ny:
                        d[k] += 0.5 * ry * (u_old[i][Ny] + u_new[i][Ny])

            # Solve A * U_new_int = d
            A = [row[:] for row in A_template]
            U_new_int = gaussian_elimination_solve(A, d)

            # Rebuild u_new
            for i in range(1, Nx):
                for j in range(1, Ny):
                    k = ij_to_k(i, j, Ny)
                    u_new[i][j] = U_new_int[k]

            # Numerical error E_n = max |u_new - u_old|
            max_diff = 0.0
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    diff = abs(u_new[i][j] - u_old[i][j])
                    if diff > max_diff:
                        max_diff = diff

            # Write snapshot
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    f_sol.write(f"{n} {t} {i} {j} {x[i]} {y[j]} {u_new[i][j]}\n")

            E_list.append(max_diff)
            t_list.append(t)

            # Next step
            u_old, u_new = u_new, u_old

    # Convergence file
    with open(output_conv, "w") as f_conv:
        f_conv.write("# 2D Heat Crank–Nicolson convergence data\n")
        f_conv.write("# col1: n\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_{i,j} |u_ij^n - u_ij^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("2D Crank–Nicolson heat solution written to:", output_sol)
    print("Convergence written to:", output_conv)
    print("Example gnuplot:")
    print("  splot 'heat2d_cn_solution.dat' using 5:6:7 every :::(Nt*(Nx+1)*(Ny+1)):: with points")
    print("  plot  'heat2d_cn_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'")

    if HAVE_MPL:
        # Final-time slice
        u_final = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]
        with open(output_sol, "r") as f:
            lines = [ln for ln in f.readlines() if not ln.startswith("#")]
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
        plt.title("2D Heat Crank–Nicolson: final temperature")

        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n")
        plt.title("2D Heat Crank–Nicolson: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()