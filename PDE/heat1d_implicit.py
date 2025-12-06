"""
1D Heat equation – implicit scheme (Backward Euler / BTCS).

PDE:
    u_t = alpha * u_xx,   x in [0, L], t in [0, T]

BTCS scheme:
    u_i^{n+1} - u_i^n = r * (u_{i+1}^{n+1} - 2 u_i^{n+1} + u_{i-1}^{n+1})

Rearranged (for interior points i=1..Nx-1 at time n+1):
    -r * u_{i-1}^{n+1} + (1 + 2r) * u_i^{n+1} - r * u_{i+1}^{n+1} = u_i^n

This leads to a tridiagonal linear system A * u^{n+1} = d at each step.

What this script does:
- Builds the tridiagonal matrix A once.
- For each time step, builds RHS d using u^n and boundary values.
- Solves A u^{n+1} = d via basic Gaussian elimination (no pivoting).
- Tracks numerical error:
      E_n = max_i |u_i^{n+1} - u_i^n|.
- Writes:
    'heat_implicit_solution.dat'
    'heat_implicit_convergence.dat'
- Optionally plots final profile and error history.

How to use in the exam:
- Step 1: Edit USER PARAMETERS: L, T, Nx, Nt, alpha.
- Step 2: Edit u_init, u_left, u_right as needed.
- Step 3: Run:
        python3 heat1d_implicit.py
- Step 4: gnuplot:
        plot 'heat_implicit_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
L = 1.0
T = 0.1
Nx = 20
Nt = 50
alpha = 1.0

output_sol = "heat_implicit_solution.dat"
output_conv = "heat_implicit_convergence.dat"
PRINT_MATRIX = False   # set True if you want to see A printed once
# ===============================================================


def u_init(x):
    return math.sin(math.pi * x)


def u_left(t):
    return 0.0


def u_right(t):
    return 0.0


def build_matrix_and_rhs_template(r, Nx):
    """
    Build the constant tridiagonal matrix A for BTCS:

    Interior unknowns: i = 1..Nx-1  -> dimension = Nx-1.

    A has:
        diag   = 1 + 2r
        off    = -r
    """
    n = Nx - 1
    A = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        A[i][i] = 1.0 + 2.0 * r
        if i > 0:
            A[i][i - 1] = -r
        if i < n - 1:
            A[i][i + 1] = -r

    if PRINT_MATRIX:
        print("BTCS coefficient matrix A:")
        for row in A:
            print("  ", row)

    return A


def gaussian_elimination_solve(A, d):
    """
    Simple Gaussian elimination (no pivoting) to solve A x = d.

    A is small and tridiagonal for exam-sized grids.
    """
    n = len(A)

    # forward elimination
    for k in range(n - 1):
        if abs(A[k][k]) < 1e-14:
            print("Warning: near-zero pivot in BTCS elimination.")
        m = A[k + 1][k] / A[k][k]
        for j in range(k, n):
            A[k + 1][j] -= m * A[k][j]
        d[k + 1] -= m * d[k]

    # back substitution
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
            print("Warning: near-zero diagonal encountered.")
            x[i] = 0.0
        else:
            x[i] = (d[i] - s) / A[i][i]

    return x


def main():
    dx = L / float(Nx)
    dt = T / float(Nt)
    r = alpha * dt / (dx * dx)
    print(f"Implicit heat scheme: r = {r:.3e} (unconditionally stable in theory)")

    # grid
    x = [i * dx for i in range(Nx + 1)]

    # initial condition
    u_old = [u_init(xi) for xi in x]
    u_new = [0.0 for _ in range(Nx + 1)]

    # apply BC at t=0
    u_old[0] = u_left(0.0)
    u_old[Nx] = u_right(0.0)

    # matrix A for interior unknowns
    A_template = build_matrix_and_rhs_template(r, Nx)

    E_list = []
    t_list = []

    with open(output_sol, "w") as f_sol:
        f_sol.write("# Heat equation implicit solution (BTCS)\n")
        f_sol.write("# col1: n (time index)\n")
        f_sol.write("# col2: t_n\n")
        f_sol.write("# col3: i (space index)\n")
        f_sol.write("# col4: x_i\n")
        f_sol.write("# col5: u_i^n\n")

        # n=0
        t = 0.0
        for i in range(Nx + 1):
            f_sol.write(f"0 {t} {i} {x[i]} {u_old[i]}\n")
        E_list.append(0.0)
        t_list.append(t)

        # time steps
        for n in range(1, Nt + 1):
            t = n * dt
            # RHS d based on u_old and boundary values at time t
            uL = u_left(t)
            uR = u_right(t)

            # interior RHS: size Nx-1
            d = [0.0 for _ in range(Nx - 1)]
            for i in range(1, Nx):
                d[i - 1] = u_old[i]

            # boundary terms
            d[0] += r * uL
            d[Nx - 2] += r * uR

            # copy A_template so we don't overwrite it
            A = [row[:] for row in A_template]
            u_interior = gaussian_elimination_solve(A, d)

            # build full u_new
            u_new[0] = uL
            u_new[Nx] = uR
            for i in range(1, Nx):
                u_new[i] = u_interior[i - 1]

            # measure numerical change
            max_diff = 0.0
            for i in range(Nx + 1):
                diff = abs(u_new[i] - u_old[i])
                if diff > max_diff:
                    max_diff = diff

            # write snapshot
            for i in range(Nx + 1):
                f_sol.write(f"{n} {t} {i} {x[i]} {u_new[i]}\n")

            E_list.append(max_diff)
            t_list.append(t)

            # prepare
            u_old, u_new = u_new, u_old

    with open(output_conv, "w") as f_conv:
        f_conv.write("# Heat equation BTCS convergence\n")
        f_conv.write("# col1: n\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_i |u^{n} - u^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("Implicit heat solution written to:", output_sol)
    print("Convergence written to:", output_conv)
    print("Example gnuplot:")
    print("  plot 'heat_implicit_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'")

    if HAVE_MPL:
        # final profile
        import matplotlib.pyplot as plt
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
        plt.title("Heat equation implicit: final profile")

        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n")
        plt.title("Heat implicit: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()