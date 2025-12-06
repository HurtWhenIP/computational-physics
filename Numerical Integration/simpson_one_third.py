"""
Simpson's 1/3 rule with numerical step-refinement error analysis.

What this script does:
- Approximates I = ∫_a^b f(x) dx using composite Simpson's 1/3 rule.
- Uses multiple N values (even) to study convergence.
- For each N:
    * I_N      : Simpson approximation.
    * error_N  : |I_N - I_prev| (numerical error between successive N).
- Writes data to 'simpson13_convergence.dat'.
- Optionally plots results with matplotlib.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * a, b      : limits of integration.
    * N_list    : list of even N values (e.g., [10, 20, 40]).
- Step 2: Change f(x) to your integrand.
- Step 3: Run:
        python3 simpson_one_third.py
- Step 4: Use gnuplot:
        plot 'simpson13_convergence.dat' using 1:3 with linespoints title 'I_N'
        plot 'simpson13_convergence.dat' using 1:4 with linespoints title 'error'

Data file columns:
    col1: N        (even number of subintervals)
    col2: h        (step size)
    col3: I_N      (Simpson 1/3 approximation)
    col4: error_N  (|I_N - I_prev|)
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = 0.0                        # Left limit
b = 12.0                        # Right limit
N_list = [10, 100, 500, 1000]      # Must all be even
output_file = "simpson13_convergence.dat"
# ===============================================================


def f(x):
    """
    Integrand f(x). Replace with the given function in the exam.
    """
    return math.pow(x,3)


def simpson_one_third(a, b, N):
    """
    Composite Simpson's 1/3 rule on [a, b] with N subintervals (N must be even).

    h = (b - a) / N

    I ≈ (h/3) * [f(x0) + f(xN) + 4*(odd indices) + 2*(even indices except endpoints)]
    """
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's 1/3 rule.")

    h = (b - a) / float(N)
    x0 = a
    xN = b

    s = f(x0) + f(xN)

    # Sum over odd indices
    for i in range(1, N, 2):
        x = a + i * h
        s += 4.0 * f(x)

    # Sum over even indices (excluding endpoints)
    for i in range(2, N, 2):
        x = a + i * h
        s += 2.0 * f(x)

    I = (h / 3.0) * s
    return I, h


def main():
    fout = open(output_file, "w")
    fout.write("# col1: N (even), col2: h, col3: I_N (Simpson 1/3), col4: error_N = |I_N - I_prev|\n")

    I_prev = None
    N_vals = []
    I_vals = []
    err_vals = []

    for N in N_list:
        I_N, h = simpson_one_third(a, b, N)
        if I_prev is None:
            error = 0.0
        else:
            error = abs(I_N - I_prev)

        fout.write(f"{N} {h} {I_N} {error}\n")

        N_vals.append(N)
        I_vals.append(I_N)
        err_vals.append(error)

        I_prev = I_N

    fout.close()

    print("Simpson's 1/3 rule convergence study finished.")
    print("Results written to:", output_file)
    print("Example gnuplot commands:")
    print("  plot 'simpson13_convergence.dat' using 1:3 with linespoints title 'I_N'")
    print("  plot 'simpson13_convergence.dat' using 1:4 with linespoints title 'error'")

    if HAVE_MPL:
        plt.figure()
        plt.plot(N_vals[1:], I_vals[1:], marker='o')
        plt.xlabel("N (even)")
        plt.ylabel("I_N (Simpson 1/3)")
        plt.title("Simpson 1/3: integral vs N")

        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]
        plt.figure()
        plt.semilogy(N_vals[1:], err_no_zero[1:], marker='o')
        plt.xlabel("N (even)")
        plt.ylabel("|I_N - I_prev|")
        plt.title("Simpson 1/3: numerical error (log scale)")

        plt.show()


if __name__ == "__main__":
    main()