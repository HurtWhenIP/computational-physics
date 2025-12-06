"""
Euler's method for solving a first-order ODE IVP: y' = f(x, y),  y(x0) = y0.

What this script does:
- Solves y' = f(x, y) on [x0, x_end] using explicit Euler method.
- Uses N steps, step size h = (x_end - x0)/N.
- Tracks a purely numerical error measure at each step:
      e_n = |y_n - y_{n-1}|
  (no analytic / true solution is used).
- Writes data to 'euler_ode_solution.dat'.
- Optionally plots y(x) and error vs x with matplotlib (if available).

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * x0, y0      : initial point and value.
    * x_end       : final x value.
    * N           : number of steps.
- Step 2: Replace f(x, y) with the given ODE's right-hand side.
- Step 3: Run:
        python3 euler_ode.py
- Step 4: gnuplot examples:
        # y vs x
        plot 'euler_ode_solution.dat' using 2:3 with linespoints title 'y(x)'
        # error vs x (skip first zero-error point)
        plot 'euler_ode_solution.dat' using 2:4 every ::1 with linespoints title 'error'

Data file:
    'euler_ode_solution.dat'
    col1: n        (step index)
    col2: x_n
    col3: y_n
    col4: e_n = |y_n - y_{n-1}| (0 for n = 0)
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
x0 = 0.0         # initial x
y0 = 1.0         # initial y = y(x0)
x_end = 1.0      # final x
N = 20           # number of steps
output_file = "euler_ode_solution.dat"
# ===============================================================


def f(x, y):
    """
    Right-hand side of the ODE: y' = f(x, y).
    Replace this with the given function in the exam.

    Example here: y' = -y  (simple exponential decay).
    """
    return -2*x + math.exp(-y)


def euler_solve(x0, y0, x_end, N, filename):
    """
    Solve y' = f(x, y) using Euler's method with N steps on [x0, x_end].

    Euler update:
        y_{n+1} = y_n + h * f(x_n, y_n)
    where h = (x_end - x0)/N.

    Also compute numerical error measure:
        e_n = |y_n - y_{n-1}|  (for n >= 1), e_0 = 0.
    """
    h = (x_end - x0) / float(N)

    # Lists to store results
    x_values = []
    y_values = []
    err_values = []

    # Initial conditions
    x = x0
    y = y0
    e = 0.0  # error at step 0 is defined as 0

    # Open output file and write header
    fout = open(filename, "w")
    fout.write("# Euler method solution for y' = f(x, y)\n")
    fout.write("# col1: n (step index)\n")
    fout.write("# col2: x_n\n")
    fout.write("# col3: y_n\n")
    fout.write("# col4: e_n = |y_n - y_{n-1}|\n")

    # Write initial point
    fout.write(f"0 {x} {y} {e}\n")
    x_values.append(x)
    y_values.append(y)
    err_values.append(e)

    # Step through the interval
    for n in range(1, N + 1):
        y_old = y
        # Euler step
        y = y + h * f(x, y)
        x = x0 + n * h
        # Numerical error = difference from previous y
        e = abs(y - y_old)

        fout.write(f"{n} {x} {y} {e}\n")
        x_values.append(x)
        y_values.append(y)
        err_values.append(e)

    fout.close()
    return x_values, y_values, err_values


def main():
    x_vals, y_vals, err_vals = euler_solve(x0, y0, x_end, N, output_file)

    print("Euler method finished.")
    print("Results written to:", output_file)
    print("Final point: x =", x_vals[-1], ", y =", y_vals[-1])
    print("Example gnuplot commands:")
    print("  plot 'euler_ode_solution.dat' using 2:3 with linespoints title 'y(x)'")
    print("  plot 'euler_ode_solution.dat' using 2:4 every ::1 with linespoints title 'error'")

    if HAVE_MPL:
        # Plot y(x)
        plt.figure()
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Euler method: y(x)")

        # Plot numerical error (skip the first zero-error point)
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]
        plt.figure()
        plt.semilogy(x_vals[1:], err_no_zero[1:], marker='o')
        plt.xlabel("x")
        plt.ylabel("|y_n - y_{n-1}|")
        plt.title("Euler method: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()