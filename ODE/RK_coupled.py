"""
RK4 for a coupled system of first-order ODEs.

Example system (simple harmonic oscillator):
    y1 = y      (position)
    y2 = v      (velocity)
    y1' = y2
    y2' = -omega^2 * y1

What this script does:
- Solves a 2D system:
      y1' = f1(x, y1, y2)
      y2' = f2(x, y1, y2)
  using RK4 on [x0, x_end] with N steps.
- Tracks a numerical error measure:
      e_n = max( |y1_n - y1_{n-1}|, |y2_n - y2_{n-1}| ).
- Writes data to 'rk4_coupled_system.dat'.
- Optionally plots y1(x), y2(x), and error.

How to use:
- Edit USER PARAMETERS:
    * x0, y1_0, y2_0, x_end, N.
    * omega in the example, or replace f1, f2 with another system.
- Run:
      python3 rk4_coupled_system.py
- gnuplot:
      plot 'rk4_coupled_system.dat' using 2:3 with linespoints title 'y1(x)'
      plot 'rk4_coupled_system.dat' using 2:4 with linespoints title 'y2(x)'
      plot 'rk4_coupled_system.dat' using 2:5 every ::1 with linespoints title 'error'
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
x0 = 0.0
y1_0 = 1.0      # initial position
y2_0 = 0.0      # initial velocity
x_end = 10.0
N = 100
omega = 1.0
output_file = "rk4_coupled_system.dat"
# ===============================================================


def f1(x, y1, y2):
    """
    First equation: y1' = f1(x, y1, y2).
    For harmonic oscillator: y1' = y2.
    """
    return y2


def f2(x, y1, y2):
    """
    Second equation: y2' = f2(x, y1, y2).
    For harmonic oscillator: y2' = -omega^2 * y1.
    """
    return -omega * omega * y1


def rk4_coupled_solve(x0, y1_0, y2_0, x_end, N, filename):
    """
    RK4 solver for 2D system (y1, y2).
    """
    h = (x_end - x0) / float(N)

    x_vals = []
    y1_vals = []
    y2_vals = []
    err_vals = []

    x = x0
    y1 = y1_0
    y2 = y2_0
    e = 0.0

    fout = open(filename, "w")
    fout.write("# RK4 for coupled system\n")
    fout.write("# col1: n\n")
    fout.write("# col2: x_n\n")
    fout.write("# col3: y1_n\n")
    fout.write("# col4: y2_n\n")
    fout.write("# col5: e_n = max(|y1_n - y1_{n-1}|, |y2_n - y2_{n-1}|)\n")

    fout.write(f"0 {x} {y1} {y2} {e}\n")
    x_vals.append(x)
    y1_vals.append(y1)
    y2_vals.append(y2)
    err_vals.append(e)

    for n in range(1, N + 1):
        y1_old = y1
        y2_old = y2

        # RK4 for system
        k1_1 = f1(x, y1, y2)
        k1_2 = f2(x, y1, y2)

        k2_1 = f1(x + 0.5 * h, y1 + 0.5 * h * k1_1, y2 + 0.5 * h * k1_2)
        k2_2 = f2(x + 0.5 * h, y1 + 0.5 * h * k1_1, y2 + 0.5 * h * k1_2)

        k3_1 = f1(x + 0.5 * h, y1 + 0.5 * h * k2_1, y2 + 0.5 * h * k2_2)
        k3_2 = f2(x + 0.5 * h, y1 + 0.5 * h * k2_1, y2 + 0.5 * h * k2_2)

        k4_1 = f1(x + h,       y1 + h * k3_1,       y2 + h * k3_2)
        k4_2 = f2(x + h,       y1 + h * k3_1,       y2 + h * k3_2)

        y1 = y1 + (h / 6.0) * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
        y2 = y2 + (h / 6.0) * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
        x = x0 + n * h

        # Numerical error as max component difference
        e = max(abs(y1 - y1_old), abs(y2 - y2_old))

        fout.write(f"{n} {x} {y1} {y2} {e}\n")
        x_vals.append(x)
        y1_vals.append(y1)
        y2_vals.append(y2)
        err_vals.append(e)

    fout.close()
    return x_vals, y1_vals, y2_vals, err_vals


def main():
    x_vals, y1_vals, y2_vals, err_vals = rk4_coupled_solve(
        x0, y1_0, y2_0, x_end, N, output_file
    )

    print("RK4 coupled system finished.")
    print("Results written to:", output_file)
    print("Final point: x =", x_vals[-1],
          ", y1 =", y1_vals[-1], ", y2 =", y2_vals[-1])

    if HAVE_MPL:
        plt.figure()
        plt.plot(x_vals, y1_vals, marker='o', label="y1 (position)")
        plt.plot(x_vals, y2_vals, marker='x', label="y2 (velocity)")
        plt.xlabel("x")
        plt.ylabel("values")
        plt.legend()
        plt.title("RK4 coupled system: y1, y2")

        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]
        plt.figure()
        plt.semilogy(x_vals[1:], err_no_zero[1:], marker='o')
        plt.xlabel("x")
        plt.ylabel("max component change")
        plt.title("RK4 coupled system: numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()