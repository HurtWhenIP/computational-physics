"""
Composite trapezoidal rule with step-refinement-based error analysis.

What this script does:
- Approximates I = ∫_a^b f(x) dx using the trapezoidal rule.
- Uses multiple values of N (number of subintervals) to study convergence.
- For each N, computes:
      I_N       : trapezoidal approximation
      error_N   : |I_N - I_prev|, where I_prev is result from previous N
- Writes data to 'trap_convergence.dat'.
- Optionally plots I_N and error_N vs N using matplotlib.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * a, b      : limits of integration.
    * N_list    : list of N values (e.g., [10, 20, 40, 80]).
    * output_file.
- Step 2: Change f(x) to your integrand.
- Step 3: Run:
        python3 trapezoidal_rule.py
- Step 4: Plot with gnuplot:
        plot 'trap_convergence.dat' using 1:3 with linespoints title 'I_N'
        plot 'trap_convergence.dat' using 1:4 with linespoints title 'error'
        set logscale y
        plot 'trap_convergence.dat' using 1:4 with linespoints title 'error (log)'

Data file columns:
    col1: N        (number of subintervals)
    col2: h        (step size)
    col3: I_N      (trapezoidal approximation)
    col4: error_N  (|I_N - I_prev|, numerical error estimate)
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = 0.0                        # Left limit of integration
b = math.pi                        # Right limit of integration
N_list = [10, 20, 40, 80, 160] # Different N values to test convergence
output_file = "trap_convergence.dat"
# ===============================================================


def f(x):
    """
    Integrand function f(x).

    In the exam:
    - Replace this with the given function.

    Example:
        f(x) = sin(x)
    """
    return math.sin(x)


def trapezoidal(a, b, N):
    """
    Composite trapezoidal rule on [a, b] with N subintervals.

    h = (b - a) / N

    I ≈ h * [0.5*f(a) + f(x1) + ... + f(x_{N-1}) + 0.5*f(b)]
    """
    h = (b - a) / float(N)
    s = 0.5 * f(a) + 0.5 * f(b)
    x = a + h
    for i in range(1, N):
        s += f(x)
        x += h
    return h * s, h


def main():
    fout = open(output_file, "w")
    fout.write("# col1: N, col2: h, col3: I_N (trap), col4: error_N = |I_N - I_prev|\n")

    I_prev = None
    N_vals = []
    I_vals = []
    err_vals = []

    for N in N_list:
        I_N, h = trapezoidal(a, b, N)
        if I_prev is None:
            error = 0.0  # first entry has no previous reference
        else:
            error = abs(I_N - I_prev)

        fout.write(f"{N} {h} {I_N} {error}\n")

        N_vals.append(N)
        I_vals.append(I_N)
        err_vals.append(error)

        I_prev = I_N

    fout.close()

    print("Trapezoidal rule convergence study finished.")
    print("Results written to:", output_file)
    print("Example gnuplot commands:")
    print("  plot 'trap_convergence.dat' using 1:3 with linespoints title 'I_N'")
    print("  plot 'trap_convergence.dat' using 1:4 with linespoints title 'error'")
    print("  set logscale y")
    print("  plot 'trap_convergence.dat' using 1:4 with linespoints title 'error (log)'")

    if HAVE_MPL:
        # Plot I_N vs N
        plt.figure()
        plt.plot(N_vals[1:], I_vals[1:], marker='o')
        plt.xlabel("N (number of subintervals)")
        plt.ylabel("I_N (trapezoidal)")
        plt.title("Trapezoidal rule: integral vs N")

        # Plot error vs N (log scale)
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]
        plt.figure()
        plt.semilogy(N_vals[1:], err_no_zero[1:], marker='o')
        plt.xlabel("N (number of subintervals)")
        plt.ylabel("|I_N - I_prev|")
        plt.title("Trapezoidal rule: numerical error vs N (log scale)")

        plt.show()


if __name__ == "__main__":
    main()