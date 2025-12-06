"""
Gaussian elimination with partial pivoting for solving A x = b.

What this script does:
- Solves a linear system A x = b using:
    1. Forward elimination with partial pivoting.
    2. Back substitution.
- Uses ONLY Python lists (no NumPy).
- Computes a residual vector r = A x - b and its Euclidean norm ||r||.
- Writes the augmented matrix [A|b] BEFORE and AFTER elimination into a .dat file.
- Prints:
    * Original matrix A and vector b.
    * Upper-triangular matrix U and modified right-hand side.
    * Solution vector x.
    * Residual norm.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * A_matrix : nested list for A (n x n).
    * b_vector : list for b (length n).
- Step 2: Run:
        python3 gaussian_elimination_pivot.py
- Step 3: gnuplot (you can inspect rows/columns if needed):
        # Example: plot the first column of A (after elimination) vs row index
        plot 'gaussian_elimination_matrix.dat' using 1:2 with linespoints title 'col1 of U'

Data file:
    'gaussian_elimination_matrix.dat'
    - Header lines start with '#'.
    - First block: original augmented matrix [A|b].
    - Second block: upper-triangular augmented matrix [U|b_mod].
    - Columns:
        col1: row index (1-based)
        col2..(n+1): columns of A (or U)
        last col: b (or modified b) as the final column.
"""

import math

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
# Example 3x3 system:
#   2x +  y -  z =  8
#  -3x - y + 2z = -11
#  -2x + y + 2z = -3
A_matrix = [
    [2.0, 1.0, -1.0],
    [-3.0, -1.0, 2.0],
    [-2.0, 1.0, 2.0]
]
b_vector = [8.0, -11.0, -3.0]

output_file = "gaussian_elimination_matrix.dat"
# ===============================================================


def print_matrix_and_vector(A, b, title):
    """
    Print the matrix A and vector b in a readable format.
    """
    print(title)
    n = len(A)
    for i in range(n):
        row_str = ""
        for j in range(len(A[i])):
            row_str += f"{A[i][j]:12.6f} "
        row_str += " | "
        row_str += f"{b[i]:12.6f}"
        print(row_str)
    print()


def forward_elimination_with_partial_pivoting(A, b):
    """
    Perform forward elimination with partial pivoting to transform [A|b]
    into an upper-triangular system [U|b_mod].

    A and b are modified IN PLACE.

    Partial pivoting:
    - For each column k, find the row p >= k where |A[p][k]| is maximum.
    - Swap rows k and p in both A and b.
    - Then eliminate entries below the pivot in column k.

    Returns:
        None (A and b are modified).
    """
    n = len(A)
    for k in range(n - 1):
        # --- Pivot selection ---
        # Find index p (>= k) such that |A[p][k]| is maximal in this column.
        pivot_row = k
        max_val = abs(A[k][k])
        for p in range(k + 1, n):
            if abs(A[p][k]) > max_val:
                max_val = abs(A[p][k])
                pivot_row = p

        # If pivot_row != k, swap rows in A and b.
        if pivot_row != k:
            A[k], A[pivot_row] = A[pivot_row], A[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]

        # If pivot is zero (or extremely close), system is singular or nearly so.
        if abs(A[k][k]) < 1.0e-14:
            print("Warning: Near-zero pivot encountered at column", k)
            continue

        # --- Elimination below pivot ---
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            # Subtract factor * row k from row i
            for j in range(k, n):  # start from k since left of k is already 0
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]


def back_substitution(U, b):
    """
    Back substitution for upper-triangular system U x = b.

    U: upper-triangular n x n matrix (list of lists).
    b: right-hand side vector of length n.

    Returns:
        x: solution vector of length n.
    """
    n = len(U)
    x = [0.0] * n

    # Start from the last row and go upwards.
    for i in range(n - 1, -1, -1):
        # Compute sum_{j=i+1..n-1} U[i][j] * x[j]
        sum_upper = 0.0
        for j in range(i + 1, n):
            sum_upper += U[i][j] * x[j]

        # Divide by the diagonal element U[i][i]
        if abs(U[i][i]) < 1.0e-14:
            print("Warning: Zero or near-zero diagonal encountered in back substitution.")
            x[i] = 0.0  # To avoid division by zero
        else:
            x[i] = (b[i] - sum_upper) / U[i][i]

    return x


def compute_residual(A_orig, b_orig, x):
    """
    Compute residual vector r = A_orig x - b_orig
    and its Euclidean norm.

    Returns:
        r: list of residuals
        norm_r: Euclidean norm of r
    """
    n = len(A_orig)
    r = [0.0] * n
    for i in range(n):
        sum_ax = 0.0
        for j in range(n):
            sum_ax += A_orig[i][j] * x[j]
        r[i] = sum_ax - b_orig[i]

    # Euclidean norm: sqrt(sum r_i^2)
    sum_sq = 0.0
    for i in range(n):
        sum_sq += r[i] * r[i]
    norm_r = math.sqrt(sum_sq)

    return r, norm_r


def write_augmented_matrix_to_dat(filename, A_orig, b_orig, A_upper, b_upper):
    """
    Write the original [A|b] and the upper-triangular [U|b_mod] to a .dat file.

    Format:
        # original matrix
        row_index  A[0][0] ... A[0][n-1]  b[0]
        ...
        blank line
        # upper-triangular matrix
        row_index  U[0][0] ... U[0][n-1]  b_mod[0]
        ...

    This is mainly for inspection; you can use gnuplot to view columns
    vs row index if needed.
    """
    n = len(A_orig)
    with open(filename, "w") as fout:
        fout.write("# Augmented matrices [A|b] and [U|b_mod]\n")
        fout.write("# First block: original [A|b]\n")
        fout.write("# col1: row index (1-based)\n")
        for j in range(n):
            fout.write(f"# col{j+2}: A column {j}\n")
        fout.write(f"# col{n+2}: b\n")

        # Original matrix
        for i in range(n):
            row_data = f"{i+1}"
            for j in range(n):
                row_data += f" {A_orig[i][j]}"
            row_data += f" {b_orig[i]}"
            fout.write(row_data + "\n")

        # Separator
        fout.write("\n# Second block: upper-triangular [U|b_mod]\n")

        # Upper-triangular matrix
        for i in range(n):
            row_data = f"{i+1}"
            for j in range(n):
                row_data += f" {A_upper[i][j]}"
            row_data += f" {b_upper[i]}"
            fout.write(row_data + "\n")


def main():
    # Make deep copies of A_matrix and b_vector so we can keep originals.
    n = len(A_matrix)
    A = []
    for i in range(n):
        A.append(A_matrix[i][:])  # copy each row
    b = b_vector[:]

    # Print original system
    print_matrix_and_vector(A, b, "Original system [A|b]:")

    # Keep a copy of the original for residual computation and output
    A_orig = []
    for i in range(n):
        A_orig.append(A_matrix[i][:])
    b_orig = b_vector[:]

    # Perform Gaussian elimination with partial pivoting
    forward_elimination_with_partial_pivoting(A, b)

    # Print the upper-triangular system
    print_matrix_and_vector(A, b, "Upper-triangular system [U|b_mod] after forward elimination:")

    # Back substitution to find solution x
    x = back_substitution(A, b)

    # Print solution
    print("Solution vector x:")
    for i in range(n):
        print(f"x[{i}] = {x[i]:12.6f}")
    print()

    # Compute and print residual
    r, norm_r = compute_residual(A_orig, b_orig, x)
    print("Residual vector r = A x - b:")
    for i in range(n):
        print(f"r[{i}] = {r[i]:12.6f}")
    print(f"\nEuclidean norm of residual ||r|| = {norm_r:.6e}\n")

    # Write matrices to .dat file
    write_augmented_matrix_to_dat(output_file, A_orig, b_orig, A, b)

    print("Augmented matrices written to:", output_file)
    print("Example gnuplot commands (e.g., view column 2 vs row index):")
    print("  plot 'gaussian_elimination_matrix.dat' using 1:2 with points title 'col1 of matrix'")
    print("  # Change the second column index to view different columns.")
    

if __name__ == "__main__":
    main()