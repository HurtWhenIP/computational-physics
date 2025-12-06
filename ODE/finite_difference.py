"""
Finite difference method for a second-order ODE BVP:

    y''(x) = r(x),   x in [a, b]
    y(a) = alpha
    y(b) = beta

Template (linear case):
- This script implements the simplest case where the ODE is:
      y''(x) = r(x)
  which leads to a linear system.
- For more general linear ODEs like:
      y'' + p(x) y' + q(x) y = r(x)
  you would modify the discrete coefficients in the matrix (see comments).

What this script does:
- Discretizes [a, b] into N intervals, h = (b - a)/N.
- Interior grid points: x_i = a + i h, i = 1..N-1.
- Uses central difference:
      y''(x_i) ≈ (y_{i-1} - 2 y_i + y_{i+1}) / h^2
- Imposes boundary conditions y_0 = alpha, y_N = beta.
- Builds a tridiagonal system A y = d for interior unknowns y_1..y_{N-1}.
- Solves it using a simple Gaussian elimination (no iterations).
- Writes solution to 'fd_bvp_solution.dat'.
- Optionally plots y(x) with matplotlib if installed.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * a, b, alpha, beta, N.
    * r_function(x) for the right-hand side.
- Step 2: Run:
        python3 finite_difference_bvp.py
- Step 3: gnuplot:
        plot 'fd_bvp_solution.dat' using 1:2 with linespoints title 'y(x)'

Data file:
    fd_bvp_solution.dat:
        col1: x_i   (all grid points, including boundaries)
        col2: y_i

How to adapt for IVP vs BVP (conceptual comments):
- BVP (this script):
    * You prescribe y(a) and y(b).
    * You discretize the entire interval and build a linear system
      for the values in between.
- IVP:
    * You prescribe y(a) and y'(a).
    * Instead of solving a global linear system, you march forward
      using an ODE solver (Euler, RK, etc.).
    * To turn this finite-difference idea into an IVP solver, you
      would not build a big matrix. Instead, you would update y_i
      step by step using something like
          y_{i+1} ≈ 2 y_i - y_{i-1} + h^2 r(x_i)
      (with y_0 and y_1 determined from initial conditions).
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = 0.0
b = 1.0
alpha = 0.0    # y(a)
beta = 0.0     # y(b)
N = 10         # number of subintervals => N+1 grid points
output_file = "fd_bvp_solution.dat"
# ===============================================================


def r_function(x):
    """
    Right-hand side r(x) in the ODE: y''(x) = r(x).

    Replace with the given function in the exam.

    Example: r(x) = -2  => y(x) = x(1 - x) satisfies y'' = -2 with y(0)=y(1)=0.
    """
    return -2.0


def build_tridiagonal_system(a, b, alpha, beta, N):
    """
    Build the tridiagonal system for y''(x) = r(x) with y(a)=alpha, y(b)=beta.

    Unknowns: y_1..y_{N-1} (interior points).
    Grid: x_i = a + i*h, i = 0..N, where h = (b-a)/N.

    Discrete equation at interior i:
        (y_{i-1} - 2 y_i + y_{i+1}) / h^2 = r(x_i)
    Rearranged:
        -2 y_i + y_{i-1} + y_{i+1} = h^2 r(x_i)

    The resulting linear system for i = 1..N-1 has:
        A[row,row-1] = 1       (sub-diagonal)
        A[row,row]   = -2      (main diagonal)
        A[row,row+1] = 1       (super-diagonal)
    with boundary conditions included in the RHS.
    """
    h = (b - a) / float(N)
    n_unknown = N - 1  # y_1..y_{N-1}

    # Initialize A (n_unknown x n_unknown) and d (rhs)
    A = [[0.0 for _ in range(n_unknown)] for _ in range(n_unknown)]
    d = [0.0 for _ in range(n_unknown)]

    for i in range(1, N):  # i = 1..N-1
        x_i = a + i * h
        row = i - 1  # index 0..n_unknown-1

        # Main diagonal
        A[row][row] = -2.0

        # Sub-diagonal
        if row - 1 >= 0:
            A[row][row - 1] = 1.0

        # Super-diagonal
        if row + 1 < n_unknown:
            A[row][row + 1] = 1.0

        # Right-hand side
        d[row] = (h * h) * r_function(x_i)

    # Incorporate boundary conditions into RHS:
    # For i = 1 (row 0): equation includes y_0 = alpha
    d[0] -= alpha
    # For i = N-1 (row n_unknown-1): equation includes y_N = beta
    d[n_unknown - 1] -= beta

    return A, d, h


def gaussian_elimination_solve(A, d):
    """
    Simple Gaussian elimination (no pivoting) for A y = d.

    Since A is tridiagonal and N is modest for exam problems,
    this is sufficient. For larger systems, a specialized
    tridiagonal solver would be more efficient.

    Returns:
        y_interior: list of y_1..y_{N-1}
    """
    n = len(A)

    # Forward elimination
    for k in range(n - 1):
        if abs(A[k][k]) < 1e-14:
            print("Warning: zero or near-zero pivot in FD Gaussian elimination.")
        m = A[k + 1][k] / A[k][k]
        for j in range(k, n):
            A[k + 1][j] -= m * A[k][j]
        d[k + 1] -= m * d[k]

    # Back substitution
    y = [0.0] * n
    if abs(A[n - 1][n - 1]) < 1e-14:
        print("Warning: zero or near-zero diagonal at last row.")
        y[n - 1] = 0.0
    else:
        y[n - 1] = d[n - 1] / A[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        sum_ax = 0.0
        for j in range(i + 1, n):
            sum_ax += A[i][j] * y[j]
        if abs(A[i][i]) < 1e-14:
            print("Warning: zero or near-zero diagonal encountered.")
            y[i] = 0.0
        else:
            y[i] = (d[i] - sum_ax) / A[i][i]

    return y


def main():
    # Build and solve the linear system for interior points
    A, d, h = build_tridiagonal_system(a, b, alpha, beta, N)
    y_interior = gaussian_elimination_solve(A, d)

    # Assemble full solution including boundaries
    xs = []
    ys = []
    for i in range(0, N + 1):
        x_i = a + i * h
        if i == 0:
            y_i = alpha
        elif i == N:
            y_i = beta
        else:
            y_i = y_interior[i - 1]
        xs.append(x_i)
        ys.append(y_i)

    # Write to file
    fout = open(output_file, "w")
    fout.write("# Finite difference BVP solution for y'' = r(x)\n")
    fout.write("# col1: x_i\n")
    fout.write("# col2: y_i\n")
    for x_i, y_i in zip(xs, ys):
        fout.write(f"{x_i} {y_i}\n")
    fout.close()

    print("Finite difference BVP finished.")
    print("Solution written to:", output_file)
    print("Example gnuplot command:")
    print("  plot 'fd_bvp_solution.dat' using 1:2 with linespoints title 'y(x)'")

    if HAVE_MPL:
        # Simple sanity plot of y(x)
        plt.figure()
        plt.plot(xs, ys, marker='o')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Finite difference BVP: y(x)")
        plt.show()


if __name__ == "__main__":
    main()