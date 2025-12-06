"""
1D Heat equation – Crank–Nicolson scheme.

PDE:
    u_t = alpha * u_xx,   x in [0, L], t in [0, T]

Crank–Nicolson (time-centered, space-centered) scheme:
    (u_i^{n+1} - u_i^n)/dt = (alpha/2) * [ (u_xx)_i^n + (u_xx)_i^{n+1} ]

Leads to linear system:
    A * u^{n+1} = B * u^n + boundary_terms

For uniform grid, r = alpha * dt / dx^2:
    Left-hand (A) tridiagonal coefficients:
        main diag: 1 + r
        off diag : -r/2
    Right-hand (B) tri-diag on u^n:
        main diag: 1 - r
        off diag :  r/2

What this script does:
- Builds matrices A and B (as dense lists-of-lists for exam-size).
- For each step:
      d = B * u_interior^n + boundary_contrib
      A * u_interior^{n+1} = d
- Tracks numerical change:
      E_n = max_i |u_i^{n+1} - u_i^n|
- Outputs:
      'heat_cn_solution.dat'
      'heat_cn_convergence.dat'
- Optional matplotlib plots.

How to use:
- Edit L, T, Nx, Nt, alpha, and IC/BC functions.
- Run:
      python3 heat1d_crank_nicolson.py
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

output_sol = "heat_cn_solution.dat"
output_conv = "heat_cn_convergence.dat"
# ===============================================================


def u_init(x):
    return math.sin(math.pi * x)


def u_left(t):
    return 0.0


def u_right(t):
    return 0.0


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
    n = len(A)
    for k in range(n - 1):
        if abs(A[k][k]) < 1e-14:
            print("Warning: near-zero pivot in CN elimination.")
        m = A[k + 1][k] / A[k][k]
        for j in range(k, n):
            A[k + 1][j] -= m * A[k][j]
        d[k + 1] -= m * d[k]

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
    print(f"Crank–Nicolson: r = {r:.3e}")

    x = [i * dx for i in range(Nx + 1)]
    u_old = [u_init(xi) for xi in x]
    u_new = [0.0 for _ in range(Nx + 1)]

    # BC at t=0
    u_old[0] = u_left(0.0)
    u_old[Nx] = u_right(0.0)

    # interior dimension
    n_int = Nx - 1

    # Build A and B
    A = [[0.0 for _ in range(n_int)] for _ in range(n_int)]
    B = [[0.0 for _ in range(n_int)] for _ in range(n_int)]

    for i in range(n_int):
        # main diag
        A[i][i] = 1.0 + r
        B[i][i] = 1.0 - r
        # sub and super diag
        if i > 0:
            A[i][i - 1] = -0.5 * r
            B[i][i - 1] = 0.5 * r
        if i < n_int - 1:
            A[i][i + 1] = -0.5 * r
            B[i][i + 1] = 0.5 * r

    E_list = []
    t_list = []

    with open(output_sol, "w") as f_sol:
        f_sol.write("# Heat equation Crank–Nicolson solution\n")
        f_sol.write("# col1: n\n")
        f_sol.write("# col2: t_n\n")
        f_sol.write("# col3: i\n")
        f_sol.write("# col4: x_i\n")
        f_sol.write("# col5: u_i^n\n")

        t = 0.0
        for i in range(Nx + 1):
            f_sol.write(f"0 {t} {i} {x[i]} {u_old[i]}\n")
        E_list.append(0.0)
        t_list.append(t)

        for n in range(1, Nt + 1):
            t = n * dt

            uL_new = u_left(t)
            uR_new = u_right(t)
            uL_old = u_left(t - dt)
            uR_old = u_right(t - dt)

            # interior vector at time n
            u_int_old = [u_old[i] for i in range(1, Nx)]

            # compute B * u_int_old
            Bu = mat_vec_mult(B, u_int_old)

            # build RHS d (size n_int)
            d = Bu[:]
            # add boundary contributions from both levels
            d[0] += 0.5 * r * (uL_old + uL_new)
            d[n_int - 1] += 0.5 * r * (uR_old + uR_new)

            # solve A * u_int_new = d
            A_copy = [row[:] for row in A]
            u_int_new = gaussian_elimination_solve(A_copy, d)

            # assemble full u_new
            u_new[0] = uL_new
            u_new[Nx] = uR_new
            for i in range(1, Nx):
                u_new[i] = u_int_new[i - 1]

            # numerical change
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

            # next step
            u_old, u_new = u_new, u_old

    with open(output_conv, "w") as f_conv:
        f_conv.write("# Heat equation Crank–Nicolson convergence\n")
        f_conv.write("# col1: n\n")
        f_conv.write("# col2: t_n\n")
        f_conv.write("# col3: E_n = max_i |u^n - u^{n-1}|\n")
        for n, (tn, En) in enumerate(zip(t_list, E_list)):
            f_conv.write(f"{n} {tn} {En}\n")

    print("Crank–Nicolson solution written to:", output_sol)
    print("Convergence written to:", output_conv)
    print("Example gnuplot:")
    print("  plot 'heat_cn_convergence.dat' using 2:3 every ::1 with linespoints title 'E_n'")

    if HAVE_MPL:
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
        plt.title("Heat equation Crank–Nicolson: final profile")

        eps = 1e-16
        E_no_zero = [E if E > 0.0 else eps for E in E_list]
        plt.figure()
        plt.semilogy(t_list[1:], E_no_zero[1:], marker='o')
        plt.xlabel("t")
        plt.ylabel("E_n")
        plt.title("Heat CN: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()