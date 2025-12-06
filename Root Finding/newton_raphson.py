"""
Newton–Raphson method for solving f(x) = 0.

What this script does:
- Implements the Newton–Raphson update:
      x_{n+1} = x_n - f(x_n) / f'(x_n)
- Lets you choose between:
      * Analytic derivative f'(x)
      * Finite-difference derivative
- Tracks purely numerical convergence:
      e_n = |x_n - x_{n-1}|
- Writes iteration data to 'newton_convergence.dat'.
- Optionally shows quick plots using matplotlib if available.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * x0        : initial guess.
    * tolerance : stopping criterion for |x_n - x_{n-1}|.
    * max_iter  : maximum iterations.
    * USE_ANALYTIC_DERIVATIVE: True or False.
    * h_fd      : step size for finite-difference derivative if needed.
- Step 2: Replace f(x) and df_analytic(x) to match the exam problem.
- Step 3: Run:
        python3 newton_raphson.py
- Step 4: Plot the convergence using gnuplot:
        plot 'newton_convergence.dat' using 1:2 with linespoints title 'x_n'
        plot 'newton_convergence.dat' using 1:4 with linespoints title 'error'

Data file columns:
    col1: iteration n
    col2: x_n
    col3: f(x_n)
    col4: e_n = |x_n - x_{n-1}|

Note (theory only, not used in code):
- When derivative is well-behaved and the initial guess is close enough,
  Newton–Raphson has quadratic convergence.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
x0 = 1.5                 # Initial guess for the root
tolerance = 1e-12        # Stop when |x_n - x_{n-1}| < tolerance
max_iter = 100           # Maximum iterations
USE_ANALYTIC_DERIVATIVE = False  # True: use df_analytic; False: use finite-difference df_numeric
h_fd = 1e-6              # Step size for finite-difference derivative
output_file = "newton_convergence.dat"
# ===============================================================


def f(x):
    """
    Function whose root you want to find: f(x) = 0.

    In the exam:
    - Replace this body with the function given in the question.
    - Make sure x0 is a reasonable starting point (close to the actual root).

    Example here:
        f(x) = x^3 - x - 2
    """
    return x * x * x - x - 2.0


def df_analytic(x):
    """
    Analytic derivative f'(x).

    In the exam:
    - If you can easily compute f'(x) by hand, put it here.
    - Then set USE_ANALYTIC_DERIVATIVE = True.

    Example corresponding to f(x) = x^3 - x - 2:
        f'(x) = 3x^2 - 1
    """
    return 3.0 * x * x - 1.0


def df_numeric(x, h):
    """
    Finite-difference approximation of f'(x).

    We use a symmetric difference formula:
        f'(x) ≈ (f(x + h) - f(x - h)) / (2h)

    This requires only function evaluations and no analytic derivative.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def newton_raphson(x0, tol, max_iter, filename):
    """
    Run the Newton–Raphson iteration starting from x0.

    Numerical convergence:
    - No "true" root is used.
    - We track e_n = |x_n - x_{n-1}| as our error measure.

    Output:
    - Writes a .dat file with columns:
        n, x_n, f(x_n), e_n
    """
    fout = open(filename, "w")
    fout.write("# col1: iter, col2: x_n, col3: f(x_n), col4: e_n = |x_n - x_{n-1}|\n")

    x_prev = x0
    x_values = [x_prev]     # store all x_n for plotting
    err_values = [0.0]      # error for the initial guess defined as 0

    for n in range(1, max_iter + 1):
        fx = f(x_prev)

        # Choose derivative method
        if USE_ANALYTIC_DERIVATIVE:
            dfx = df_analytic(x_prev)
        else:
            dfx = df_numeric(x_prev, h_fd)

        if dfx == 0.0:
            print("Derivative became zero; cannot proceed further.")
            break

        # Newton–Raphson update
        x_new = x_prev - fx / dfx

        # Numerical error estimate
        error = abs(x_new - x_prev)

        fout.write(f"{n} {x_new} {f(x_new)} {error}\n")
        x_values.append(x_new)
        err_values.append(error)

        # Stopping condition based on change in x
        if error < tol:
            x_prev = x_new
            break

        x_prev = x_new

    fout.close()

    print("Newton–Raphson finished.")
    print("Approximate root:", x_prev)
    print("Iterations used:", n)
    print("Convergence data written to:", filename)
    print("Example gnuplot commands:")
    print("  plot 'newton_convergence.dat' using 1:2 with linespoints title 'x_n'")
    print("  plot 'newton_convergence.dat' using 1:4 with linespoints title 'error'")

    # Optional matplotlib plots (if available)
    if HAVE_MPL:
        # Plot x_n vs iteration
        plt.figure()
        plt.plot(range(len(x_values) - 1), x_values[1:], marker='o')
        plt.xlabel("Iteration n")
        plt.ylabel("x_n")
        plt.title("Newton–Raphson: iterates x_n")

        # Plot error vs iteration (log scale)
        plt.figure()
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_values]
        plt.semilogy(range(len(err_no_zero) - 1), err_no_zero[1:], marker='o')
        plt.xlabel("Iteration n")
        plt.ylabel("e_n = |x_n - x_{n-1}|")
        plt.title("Newton–Raphson: numerical error (log scale)")

        plt.show()

    return x_prev


if __name__ == "__main__":
    newton_raphson(x0, tolerance, max_iter, output_file)