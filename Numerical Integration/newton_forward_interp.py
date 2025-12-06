"""
Newton's forward interpolation with numerical error from increasing number of points.

What this script does:
- Takes equally spaced tabulated data (x_i, y_i).
- Builds the forward difference table.
- Constructs Newton's forward interpolating polynomial.
- Evaluates the interpolation at a target x_eval using:
      P_m(x_eval)  with first m data points (m = 2, 3, ..., N)
- For m = 2, 3, ..., N:
    * P_m(x_eval)    : interpolation using first m points.
    * error_m        : |P_m - P_prev|, purely numerical difference between
                       successive m values.
- Writes data to 'newton_forward_convergence.dat'.
- Optionally plots convergence with matplotlib if available.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * x_values, y_values : tabulated data (must be equally spaced in x).
    * x_eval             : the point at which you want to interpolate.
- Step 2: Ensure x_values are equally spaced.
- Step 3: Run:
        python3 newton_forward_interpolation.py
- Step 4: Use gnuplot:
        plot 'newton_forward_convergence.dat' using 1:2 with linespoints title 'P_m'
        plot 'newton_forward_convergence.dat' using 1:3 with linespoints title 'error'

Data file columns:
    col1: m          (number of data points used, from 2 to N)
    col2: P_m        (interpolated value using first m points)
    col3: error_m    (|P_m - P_prev|, purely numerical)

Note:
- There is no analytic "true" interpolation value used.
- Error is based only on the change in interpolation when one more point is added.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
# Example data: f(x) = sin(x) at equally spaced points
x_values = [0.0, 0.25 * math.pi, 0.5 * math.pi, 0.75 * math.pi, math.pi]
y_values = [math.sin(x) for x in x_values]

# Point where we want to evaluate the interpolating polynomial
x_eval = 0.4 * math.pi

output_file = "newton_forward_convergence.dat"
# ===============================================================


def build_forward_difference_table(x_vals, y_vals):
    """
    Build the forward difference table for Newton's forward interpolation.

    The table is stored as a list of lists:
        diff_table[0] = [Δ^0 y_0, Δ^0 y_1, ..., Δ^0 y_{n-1}]  = original y values
        diff_table[1] = [Δ^1 y_0, Δ^1 y_1, ..., Δ^1 y_{n-2}]
        diff_table[2] = [Δ^2 y_0, Δ^2 y_1, ..., Δ^2 y_{n-3}]
        ...
    """
    n = len(x_vals)
    # Initialize first row with the original y values.
    diff_table = []
    diff_table.append(y_vals[:])  # copy

    # Compute higher order differences
    for order in range(1, n):
        prev = diff_table[order - 1]
        current = []
        # Δ^k y_i = Δ^{k-1} y_{i+1} - Δ^{k-1} y_i
        for i in range(len(prev) - 1):
            current.append(prev[i + 1] - prev[i])
        diff_table.append(current)

    return diff_table


def newton_forward_value(x_vals, diff_table, x_eval, m):
    """
    Evaluate Newton's forward interpolating polynomial at x_eval
    using the first m data points.

    Inputs:
        x_vals     : list of x_i (equally spaced).
        diff_table : forward difference table (as built above).
        x_eval     : point where interpolation is needed.
        m          : number of data points used (2 <= m <= n).

    Steps:
    1. Compute step size h = x_1 - x_0.
    2. Compute u = (x_eval - x_0) / h.
    3. Use the Newton forward formula:
           P(x) = y_0
                  + u      * Δ^1 y_0
                  + u(u-1)/2! * Δ^2 y_0
                  + u(u-1)(u-2)/3! * Δ^3 y_0
                  + ...
       up to order (m-1).
    """
    h = x_vals[1] - x_vals[0]
    u = (x_eval - x_vals[0]) / h

    # Start with y_0
    result = diff_table[0][0]

    # Factorial accumulator and u product
    factorial = 1.0
    u_term = 1.0

    # Add higher order terms up to order (m-1)
    for k in range(1, m):
        factorial *= k
        u_term *= (u - (k - 1))
        term = (u_term / factorial) * diff_table[k][0]
        result += term

    return result


def main():
    n = len(x_values)
    if len(y_values) != n:
        print("Error: x_values and y_values must have the same length.")
        return

    # Build the forward difference table once.
    diff_table = build_forward_difference_table(x_values, y_values)

    fout = open(output_file, "w")
    fout.write("# col1: m (number of points used, from 2 to N)\n")
    fout.write("# col2: P_m(x_eval) (interpolation using first m points)\n")
    fout.write("# col3: error_m = |P_m - P_prev| (purely numerical)\n")
    fout.write(f"# x_eval = {x_eval}\n")

    m_values = []
    P_values = []
    err_values = []

    P_prev = None

    # m goes from 2 to n, using more and more points.
    for m in range(2, n + 1):
        P_m = newton_forward_value(x_values, diff_table, x_eval, m)

        if P_prev is None:
            error = 0.0
        else:
            error = abs(P_m - P_prev)

        fout.write(f"{m} {P_m} {error}\n")

        m_values.append(m)
        P_values.append(P_m)
        err_values.append(error)

        P_prev = P_m

    fout.close()

    print("Newton forward interpolation study finished.")
    print("Results written to:", output_file)
    print("Interpolation point x_eval =", x_eval)
    print("Last interpolated value P_m =", P_prev)
    print("Example gnuplot commands:")
    print("  plot 'newton_forward_convergence.dat' using 1:2 with linespoints title 'P_m'")
    print("  plot 'newton_forward_convergence.dat' using 1:3 with linespoints title 'error'")

    if HAVE_MPL:
        # Plot P_m vs m
        plt.figure()
        plt.plot(m_values, P_values, marker='o')
        plt.xlabel("m (number of points used)")
        plt.ylabel("P_m(x_eval)")
        plt.title("Newton forward interpolation: value vs m")

        # Log-plot for numerical error
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_values]

        plt.figure()
        plt.semilogy(m_values, err_no_zero, marker='o')
        plt.xlabel("m (number of points used)")
        plt.ylabel("|P_m - P_prev|")
        plt.title("Newton forward interpolation: numerical error (log scale)")

        plt.show()


if __name__ == "__main__":
    main()