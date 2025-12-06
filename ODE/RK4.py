"""
Fourth-order Runge–Kutta (RK4) method for y' = f(x, y).

What this script does:
- Solves y' = f(x, y) on [x0, x_end] using classical RK4:
      k1 = f(x_n, y_n)
      k2 = f(x_n + h/2, y_n + (h/2) * k1)
      k3 = f(x_n + h/2, y_n + (h/2) * k2)
      k4 = f(x_n + h,   y_n + h * k3)
      y_{n+1} = y_n + (h/6) * (k1 + 2k2 + 2k3 + k4)
- Tracks numerical error e_n = |y_n - y_{n-1}|.
- Writes data to 'rk4_ode_solution.dat'.
- Optionally plots y(x) and error.

How to use:
- Edit x0, y0, x_end, N and f(x, y).
- Run:
      python3 rk4_ode.py
- gnuplot:
      plot 'rk4_ode_solution.dat' using 2:3 with linespoints title 'y(x)'
      plot 'rk4_ode_solution.dat' using 2:4 every ::1 with linespoints title 'error'
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
x0 = 0.0
y0 = 1.0
x_end = 1.0
N = 20
output_file = "rk4_ode_solution.dat"
# ===============================================================


def f(x, y):
    """
    Right-hand side y' = f(x, y). Replace with your ODE.
    """
    return -y


def rk4_solve(x0, y0, x_end, N, filename):
    """
    Classical fourth-order Runge–Kutta solver.
    """
    h = (x_end - x0) / float(N)

    x_vals = []
    y_vals = []
    err_vals = []

    x = x0
    y = y0
    e = 0.0

    fout = open(filename, "w")
    fout.write("# RK4 method solution\n")
    fout.write("# col1: n\n")
    fout.write("# col2: x_n\n")
    fout.write("# col3: y_n\n")
    fout.write("# col4: e_n = |y_n - y_{n-1}|\n")

    fout.write(f"0 {x} {y} {e}\n")
    x_vals.append(x)
    y_vals.append(y)
    err_vals.append(e)

    for n in range(1, N + 1):
        y_old = y

        k1 = f(x, y)
        k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(x + h,       y + h * k3)

        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x = x0 + n * h

        e = abs(y - y_old)

        fout.write(f"{n} {x} {y} {e}\n")
        x_vals.append(x)
        y_vals.append(y)
        err_vals.append(e)

    fout.close()
    return x_vals, y_vals, err_vals


def main():
    x_vals, y_vals, err_vals = rk4_solve(x0, y0, x_end, N, output_file)

    print("RK4 method finished.")
    print("Results written to:", output_file)
    print("Final point: x =", x_vals[-1], ", y =", y_vals[-1])

    if HAVE_MPL:
        plt.figure()
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("RK4 method: y(x)")

        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]
        plt.figure()
        plt.semilogy(x_vals[1:], err_no_zero[1:], marker='o')
        plt.xlabel("x")
        plt.ylabel("|y_n - y_{n-1}|")
        plt.title("RK4 method: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()