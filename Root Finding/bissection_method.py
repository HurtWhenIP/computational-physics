"""
Bisection method for solving f(x) = 0 on an interval [a, b].

What this script does:
- Uses the bisection method to approximate a root of f(x).
- Tracks numerical convergence using e_n = |x_n - x_{n-1}| (purely numerical).
- Writes iteration data to 'bisection_convergence.dat' for plotting in gnuplot.
- Optionally shows quick plots using matplotlib if it is installed.

How to use in the exam:
- Step 1: Edit the USER PARAMETERS section:
    * a, b       : endpoints of the initial interval (must have f(a)*f(b) < 0).
    * tolerance  : desired accuracy.
    * max_iter   : maximum iterations allowed.
    * output_file: name of the .dat file (you can keep the default).
- Step 2: Change the function f(x) to match the exam question.
- Step 3: Run in terminal:
        python3 bisection_method.py
- Step 4: Use gnuplot with the output file:
        plot 'bisection_convergence.dat' using 1:2 with linespoints title 'x_n'
        plot 'bisection_convergence.dat' using 1:4 with linespoints title 'error'

Data file columns:
    col1: iteration number k
    col2: x_k (midpoint at iteration k)
    col3: f(x_k)
    col4: e_k = |x_k - x_{k-1}| (numerical error estimate)

Note:
- The classical error bound and order of convergence can be mentioned in your answer
  as theory, but here we only use purely numerical differences.
"""

import math

# Try to import matplotlib for quick sanity plots (OPTIONAL).
# In the exam environment, if matplotlib is not available, the script still works.
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = 1.0           # Left endpoint of interval [a, b]
b = 2.0           # Right endpoint of interval [a, b]
tolerance = 1e-6  # Stop when |x_n - x_{n-1}| or |b - a| < tolerance
max_iter = 100    # Safety cap on iterations
output_file = "bisection_convergence.dat"
# ===============================================================


def f(x):
    """
    Function whose root you want to find: solve f(x) = 0.

    In the exam:
    - Replace this with the required function.
    - Make sure that f(a) * f(b) < 0 for the chosen interval [a, b].

    Example here (you will change this):
        f(x) = x^3 - x - 2
    """
    return x * x * x - x - 2.0


def bisection(a, b, tol, max_iter, filename):
    """
    Perform the bisection method on [a, b] with the given tolerance.

    Numerical convergence:
    - We do NOT use the true root.
    - We only use the difference between successive midpoints:

        e_k = |x_k - x_{k-1}|

    Output:
    - Writes lines to 'filename' with columns:
        iter, x_mid, f(x_mid), e_k
    - Returns the final midpoint as the approximate root.
    """
    fa = f(a)
    fb = f(b)

    # Check sign change condition, otherwise bisection is not guaranteed to work.
    if fa * fb > 0.0:
        print("Error: f(a) and f(b) must have opposite signs for bisection.")
        return None

    fout = open(filename, "w")
    fout.write("# col1: iter, col2: x_mid, col3: f(x_mid), col4: e_k = |x_k - x_{k-1}|\n")

    x_prev = None          # previous midpoint (for error estimate)
    x_values = []          # store x_k for plotting
    err_values = []        # store e_k for plotting

    for k in range(1, max_iter + 1):
        # 1. Compute midpoint of the current interval
        mid = 0.5 * (a + b)
        fmid = f(mid)

        # 2. Compute numerical error estimate (difference from previous iterate)
        if x_prev is None:
            error = 0.0   # no previous point in the first iteration
        else:
            error = abs(mid - x_prev)

        # 3. Store data in the .dat file and in memory
        fout.write(f"{k} {mid} {fmid} {error}\n")
        x_values.append(mid)
        err_values.append(error)

        # 4. Check stopping conditions
        #    (a) interval length small enough OR
        #    (b) change in x small enough
        if abs(b - a) < tol or (x_prev is not None and error < tol):
            x_prev = mid
            break

        # 5. Bisection update:
        #    If f(a) and f(mid) have opposite signs, the root is in [a, mid],
        #    otherwise it is in [mid, b].
        if fa * fmid < 0.0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

        # 6. Store current midpoint as previous for next iteration
        x_prev = mid

    fout.close()

    print("Bisection finished.")
    print("Approximate root:", x_prev)
    print("Iterations used:", k)
    print("Convergence data written to:", filename)
    print("Example gnuplot commands:")
    print("  plot 'bisection_convergence.dat' using 1:2 with linespoints title 'x_n'")
    print("  plot 'bisection_convergence.dat' using 1:4 with linespoints title 'error'")

    # Optional matplotlib plots (sanity check; safe if matplotlib is missing)
    if HAVE_MPL:
        # Plot x_k vs iteration
        plt.figure()
        plt.plot(range(1, len(x_values)), x_values[1:], marker='o')
        plt.xlabel("Iteration k")
        plt.ylabel("Midpoint x_k")
        plt.title("Bisection method: iterates x_k")

        # Plot error vs iteration in log scale (skip zero error by adding epsilon)
        plt.figure()
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_values]
        plt.semilogy(range(1, len(err_no_zero)), err_no_zero[1:], marker='o')
        plt.xlabel("Iteration k")
        plt.ylabel("e_k = |x_k - x_{k-1}|")
        plt.title("Bisection method: numerical error (log scale)")

        plt.show()

    return x_prev


if __name__ == "__main__":
    bisection(a, b, tolerance, max_iter, output_file)