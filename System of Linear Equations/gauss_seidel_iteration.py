"""
Gauss–Seidel iterative method for solving A x = b.

What this script does:
- Uses Gauss–Seidel iteration to solve A x = b.
- Uses ONLY Python lists (no NumPy).
- Tracks convergence using:
      e_k = || x^(k) - x^(k-1) ||_2
  (purely numerical, no reference to exact solution).
- Stops when:
    e_k < tolerance  OR  iteration count reaches max_iter.
- Writes iteration data to 'gauss_seidel_convergence.dat'.
- Optionally uses matplotlib to plot:
    * error vs iteration (log scale).

How Gauss–Seidel differs from Jacobi:
- Gauss–Seidel updates x_i^(k) in-place as soon as it is computed,
  so later components in the same iteration use the newest values.
- In comments only: typically converges faster than Jacobi if convergence conditions are met.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * A_matrix   : nested list for A (n x n).
    * b_vector   : list for b.
    * x0         : initial guess (list length n).
    * tolerance  : stopping threshold on ||x^(k) - x^(k-1)||.
    * max_iter   : maximum number of iterations allowed.
- Step 2: Run:
        python3 gauss_seidel_iteration.py
- Step 3: gnuplot:
        plot 'gauss_seidel_convergence.dat' using 1:(column_with_error) with linespoints

Data file:
    'gauss_seidel_convergence.dat'
    Columns:
        col1: iteration k (starting from 0)
        col2..(n+1): x^(k)_0, x^(k)_1, ..., x^(k)_{n-1}
        last col: error e_k = ||x^(k) - x^(k-1)|| (0 for k=0)
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
# Example system (same as before):
A_matrix = [
    [10.0, 1.0, 1.0],
    [2.0, 10.0, 1.0],
    [2.0, 2.0, 10.0]
]
b_vector = [12.0, 13.0, 14.0]

# Initial guess for x
x0 = [0.0, 0.0, 0.0]

tolerance = 1e-6
max_iter = 100

output_file = "gauss_seidel_convergence.dat"
# ===============================================================


def gauss_seidel_iteration(A, b, x_initial, tol, max_iter, filename):
    """
    Perform Gauss–Seidel iteration to solve A x = b.

    Inputs:
        A         : coefficient matrix (n x n) as list of lists.
        b         : right-hand side vector (length n).
        x_initial : initial guess vector (length n).
        tol       : tolerance for ||x^(k) - x^(k-1)||.
        max_iter  : maximum number of iterations.
        filename  : name of .dat file to store convergence data.

    Output:
        x         : approximate solution.
        k         : number of iterations performed.
    """
    n = len(A)
    x_old = x_initial[:]
    x_new = x_initial[:]  # will be overwritten in-place

    # Open data file and write header
    fout = open(filename, "w")
    fout.write("# Gauss–Seidel iteration convergence data\n")
    fout.write("# col1: iteration k\n")
    for i in range(n):
        fout.write(f"# col{i+2}: x_{i}^(k)\n")
    fout.write(f"# col{n+2}: error e_k = ||x^(k) - x^(k-1)||_2 (0 for k=0)\n")

    # Helper function for Euclidean norm of difference
    def diff_norm(xa, xb):
        s = 0.0
        for i in range(n):
            d = xa[i] - xb[i]
            s += d * d
        return math.sqrt(s)

    # Write initial guess (iteration 0)
    error = 0.0
    fout.write("0")
    for i in range(n):
        fout.write(f" {x_old[i]}")
    fout.write(f" {error}\n")

    for k in range(1, max_iter + 1):
        # Copy previous iterate to check error later
        for i in range(n):
            x_new[i] = x_old[i]

        # Gauss–Seidel update:
        # x_i^(k) = (1 / a_ii) * (b_i - sum_{j < i} a_ij * x_j^(k) - sum_{j > i} a_ij * x_j^(k-1))
        for i in range(n):
            # Sum over j < i with newest values (x_new)
            sum_lower = 0.0
            for j in range(i):
                sum_lower += A[i][j] * x_new[j]

            # Sum over j > i with old values (x_old)
            sum_upper = 0.0
            for j in range(i + 1, n):
                sum_upper += A[i][j] * x_old[j]

            if abs(A[i][i]) < 1.0e-14:
                print("Warning: Zero or near-zero diagonal element in Gauss–Seidel method.")
                # Keep previous value to avoid division by zero
                x_new[i] = x_old[i]
            else:
                x_new[i] = (b[i] - sum_lower - sum_upper) / A[i][i]

        # Compute numerical error e_k = ||x_new - x_old||
        error = diff_norm(x_new, x_old)

        # Write this iteration to file
        fout.write(f"{k}")
        for i in range(n):
            fout.write(f" {x_new[i]}")
        fout.write(f" {error}\n")

        # Check convergence
        if error < tol:
            print(f"Gauss–Seidel converged after {k} iterations with error {error:.6e}.")
            x_old = x_new[:]
            break

        # Prepare for next iteration
        x_old = x_new[:]

    fout.close()

    return x_old, k, error


def main():
    x_approx, iters_used, final_error = gauss_seidel_iteration(
        A_matrix, b_vector, x0, tolerance, max_iter, output_file
    )

    print("\nApproximate solution from Gauss–Seidel iteration:")
    for i in range(len(x_approx)):
        print(f"x[{i}] = {x_approx[i]:12.6f}")
    print(f"Iterations used: {iters_used}")
    print(f"Final error e_k: {final_error:.6e}")
    print("Convergence data written to:", output_file)
    print("Example gnuplot command (for error vs iteration):")
    print("  plot 'gauss_seidel_convergence.dat' using 1:4 with linespoints title 'error'")

    if HAVE_MPL:
        # Read data back for plotting
        it_list = []
        err_list = []
        with open(output_file, "r") as fin:
            for line in fin:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                k = int(parts[0])
                e_k = float(parts[-1])
                it_list.append(k)
                err_list.append(e_k)

        # Plot error vs iteration (log scale)
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_list]
        plt.figure()
        plt.semilogy(it_list[1:], err_no_zero[1:], marker='o')
        plt.xlabel("Iteration k")
        plt.ylabel("e_k = ||x^(k) - x^(k-1)||_2")
        plt.title("Gauss–Seidel method: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()