"""
Simpson's 3/8 rule with numerical step-refinement error analysis.

What this script does:
- Approximates I = ∫_a^b f(x) dx using composite Simpson's 3/8 rule.
- Uses multiple N values (N must be a multiple of 3) to study convergence.
- For each N:
    * I_N      : Simpson 3/8 approximation.
    * error_N  : |I_N - I_prev|, where I_prev is the approximation for the
                 previous N in N_list (purely numerical error estimate).
- Writes data to 'simpson38_convergence.dat'.
- Optionally plots results with matplotlib if available.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * a, b      : limits of integration.
    * N_list    : list of N values (all must be multiples of 3),
                  for example [3, 6, 12, 24].
- Step 2: Change f(x) to your integrand.
- Step 3: Run:
        python3 simpson_three_eighth.py
- Step 4: Use gnuplot:
        plot 'simpson38_convergence.dat' using 1:3 with linespoints title 'I_N'
        plot 'simpson38_convergence.dat' using 1:4 with linespoints title 'error'

Data file columns:
    col1: N        (number of subintervals, multiple of 3)
    col2: h        (step size)
    col3: I_N      (Simpson 3/8 approximation)
    col4: error_N  (|I_N - I_prev|, purely numerical)

Note:
- Simpson's 3/8 rule is another member of the Simpson family.
- It uses cubic polynomials over groups of 3 subintervals.
"""

import math

# Try to import matplotlib for sanity-check plots (optional).
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = 0.0                          # Left limit of integration
b = math.pi                      # Right limit of integration
N_list = [3, 6, 12, 24, 48]      # All must be multiples of 3
output_file = "simpson38_convergence.dat"
# ===============================================================


def f(x):
    """
    Integrand f(x). Replace this with the given function in the exam.

    Example here: f(x) = sin(x)
    """
    return math.sin(x)


def simpson_three_eighth(a, b, N):
    """
    Composite Simpson's 3/8 rule on [a, b] with N subintervals.
    N must be a multiple of 3.

    Step-size:
        h = (b - a) / N

    Formula (composite):
        I ≈ (3h/8) * [f(x0) + f(xN)
                      + 3 * sum(f(x_i) for i = 1,2,4,5,7,8,... not multiples of 3)
                      + 2 * sum(f(x_i) for i = 3,6,9,...,N-3)]
    where:
        x_i = a + i*h
    """
    if N % 3 != 0:
        raise ValueError("N must be a multiple of 3 for Simpson's 3/8 rule.")

    h = (b - a) / float(N)

    # x0 and xN are the endpoints
    x0 = a
    xN = b

    # Start with f(x0) + f(xN)
    s = f(x0) + f(xN)

    # Sum for indices i = 1, 2, 4, 5, 7, 8, ... (i not multiple of 3)
    sum3 = 0.0  # will be multiplied by 3
    sum2 = 0.0  # will be multiplied by 2 for multiples of 3 (excluding endpoints)

    for i in range(1, N):
        x = a + i * h
        if i % 3 == 0:
            # Indices that are multiples of 3 (3, 6, 9, ..., N-3)
            sum2 += f(x)
        else:
            # All other interior points
            sum3 += f(x)

    I = (3.0 * h / 8.0) * (s + 3.0 * sum3 + 2.0 * sum2)
    return I, h


def main():
    # Open output file for convergence data.
    fout = open(output_file, "w")
    fout.write("# col1: N (multiple of 3)\n")
    fout.write("# col2: h (step size)\n")
    fout.write("# col3: I_N (Simpson 3/8 approximation)\n")
    fout.write("# col4: error_N = |I_N - I_prev|, purely numerical\n")

    I_prev = None    # Will store previous approximation
    N_vals = []      # For plotting
    I_vals = []      # For plotting
    err_vals = []    # For plotting

    # Loop over all N values to study convergence.
    for N in N_list:
        I_N, h = simpson_three_eighth(a, b, N)

        # Numerical error estimate based on difference with previous N.
        if I_prev is None:
            error = 0.0
        else:
            error = abs(I_N - I_prev)

        # Write one line per N to the data file.
        fout.write(f"{N} {h} {I_N} {error}\n")

        # Save for optional matplotlib plots.
        N_vals.append(N)
        I_vals.append(I_N)
        err_vals.append(error)

        # Update previous value.
        I_prev = I_N

    fout.close()

    # Print a brief summary to the terminal.
    print("Simpson's 3/8 rule convergence study finished.")
    print("Results written to:", output_file)
    print("Last approximation I_N =", I_prev)
    print("Example gnuplot commands:")
    print("  plot 'simpson38_convergence.dat' using 1:3 with linespoints title 'I_N'")
    print("  plot 'simpson38_convergence.dat' using 1:4 with linespoints title 'error'")

    # Optional sanity-check plots using matplotlib, if available.
    if HAVE_MPL:
        # Plot integral approximation vs N.
        plt.figure()
        plt.plot(N_vals[1:], I_vals[1:], marker='o')
        plt.xlabel("N (multiple of 3)")
        plt.ylabel("I_N (Simpson 3/8)")
        plt.title("Simpson 3/8: integral vs N")

        # Plot numerical error on a log scale.
        # Replace zero errors with a tiny positive number to avoid log(0).
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]

        plt.figure()
        plt.semilogy(N_vals[1:], err_no_zero[1:], marker='o')
        plt.xlabel("N (multiple of 3)")
        plt.ylabel("|I_N - I_prev|")
        plt.title("Simpson 3/8: numerical error (log scale)")

        plt.show()


if __name__ == "__main__":
    main()