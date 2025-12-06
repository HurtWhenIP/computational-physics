"""
Second-order Runge–Kutta method (generic RK2 template).

This script:
- Solves the IVP: y' = f(x, y),  y(x0) = y0.
- Uses the generic RK2 formulation:
        k1 = h f(x_n, y_n)
        k2 = h f(x_n + a2*h, y_n + b21*k1)
        y_{n+1} = y_n + c1*k1 + c2*k2
- Writes (x_n, y_n, numerical_error) to 'rk2_solution.dat'.
- Optionally plots using matplotlib.

IMPORTANT:
Only ONE RK2 variant is implemented and used in code.
But the two common special cases (Heun and Midpoint) are included
below as comments for the exam.

--------------------------------------------------------------------
HEUN’S METHOD (Improved Euler)  →  RK2 with:
    a2  = 1
    b21 = 1
    c1  = 1/2
    c2  = 1/2
This gives:
    k1 = h f(x_n, y_n)
    k2 = h f(x_n + h, y_n + k1)
    y_{n+1} = y_n + (k1 + k2)/2

--------------------------------------------------------------------
MIDPOINT METHOD  →  RK2 with:
    a2  = 1/2
    b21 = 1/2
    c1  = 0
    c2  = 1
This gives:
    k1 = h f(x_n, y_n)
    k2 = h f(x_n + h/2, y_n + k1/2)
    y_{n+1} = y_n + k2

--------------------------------------------------------------------
GENERIC RK2 (the form actually implemented in code):
You set (a2, b21, c1, c2) in USER PARAMETERS.

How to use in the exam:
1. Edit USER PARAMETERS block.
2. Replace f(x, y) with the required derivative.
3. Run:
       python3 rk2_method.py
4. gnuplot:
       plot 'rk2_solution.dat' using 1:2 with linespoints title 'y(x)'
       plot 'rk2_solution.dat' using 1:3 every ::1 with linespoints title 'error'
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
x0 = 0.0          # initial x
y0 = 1.0          # initial y
xf = 1.0          # final x
N  = 20           # number of steps
output_file = "rk2_solution.dat"

# Choose RK2 coefficients (generic RK2):
# For Heun:      a2 = 1.0,   b21 = 1.0,   c1 = 0.5, c2 = 0.5
# For Midpoint:  a2 = 0.5,   b21 = 0.5,   c1 = 0.0, c2 = 1.0
a2  = 0.5
b21 = 0.5
c1  = 0.0
c2  = 1.0
# ===============================================================


def f(x, y):
    """Derivative function y' = f(x, y). Edit this in the exam."""
    return -2.0 * y + math.sin(x)


def rk2_step(x, y, h):
    """
    One RK2 step using generic coefficients (a2, b21, c1, c2):
        k1 = h f(x, y)
        k2 = h f(x + a2*h, y + b21*k1)
        y_next = y + c1*k1 + c2*k2
    """
    k1 = h * f(x, y)
    k2 = h * f(x + a2 * h, y + b21 * k1)
    y_next = y + c1 * k1 + c2 * k2
    return y_next, k1, k2


def main():
    h = (xf - x0) / float(N)

    xs = []
    ys = []
    errs = []

    x = x0
    y = y0

    xs.append(x)
    ys.append(y)
    errs.append(0.0)   # first point has no previous step

    for i in range(N):
        y_next, k1, k2 = rk2_step(x, y, h)
        err = abs(y_next - y)   # purely numerical

        x = x + h
        y = y_next

        xs.append(x)
        ys.append(y)
        errs.append(err)

    # Write output
    fout = open(output_file, "w")
    fout.write("# RK2 solution\n")
    fout.write("# col1: x\n")
    fout.write("# col2: y\n")
    fout.write("# col3: error = |y_n - y_{n-1}| (skip first)\n")
    for xi, yi, ei in zip(xs, ys, errs):
        fout.write(f"{xi} {yi} {ei}\n")
    fout.close()

    print("RK2 completed.")
    print("Results saved to:", output_file)
    print("Example gnuplot commands:")
    print("  plot 'rk2_solution.dat' using 1:2 with linespoints title 'y(x)'")
    print("  plot 'rk2_solution.dat' using 1:3 every ::1 with linespoints title 'error'")

    if HAVE_MPL:
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("RK2 solution")

        # Skip first zero-error point
        eps = 1e-16
        err_no_zero = [e if e > 0 else eps for e in errs]

        plt.figure()
        plt.semilogy(xs[1:], err_no_zero[1:], marker="o")
        plt.xlabel("x")
        plt.ylabel("error")
        plt.title("RK2 numerical error (log scale)")
        plt.show()


if __name__ == "__main__":
    main()