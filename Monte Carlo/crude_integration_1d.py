"""
Crude Monte Carlo integration in 1D with numerical error analysis.

Goal:
    Approximate the integral
        I = ∫_a^b f(x) dx

Method (crude Monte Carlo):
    - Draw X_i ~ Uniform(a, b).
    - Compute Y_i = f(X_i).
    - Estimate:
          mean_f(N) = (1/N) Σ Y_i
          I_N       = (b - a) * mean_f(N)
    - Estimate variance of the estimator numerically:
          s_f^2(N)  = sample variance of Y_i
          Var[I_N] ≈ (b - a)^2 * s_f^2(N) / N

Numerical error (NO true integral used):
    - Use successive estimates:
          error_N = |I_N - I_prev|
      where I_prev is the estimate at the previous N in N_list.

Outputs:
    - 'mc_crude1d.dat'

        col1: N          (number of samples)
        col2: I_N        (current MC estimate of the integral)
        col3: Var_I_N    (numerical variance estimate of I_N)
        col4: error_N    (|I_N - I_prev|)

How to use in the exam:
    - Step 1: Set USER PARAMETERS:
          a, b, N_list, and edit f(x) to the given integrand.
    - Step 2: Run:
          python3 mc_crude_integration_1d.py
    - Step 3: In gnuplot:
          plot 'mc_crude1d.dat' using 1:2 with linespoints title 'I_N'
          plot 'mc_crude1d.dat' using 1:4 with linespoints title 'error'

Matplotlib:
    - If available, plots:
        * I_N vs N
        * error_N vs N (semilog, skipping the first N where error_N = 0)
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = 0.0
b = 1.0
N_list = [100, 500, 1000, 5000]
output_file = "mc_crude1d.dat"

# LCG parameters
seed = 12345
A    = 1103515245
C    = 12345
M    = 2**31 - 1
# ===============================================================


class LCG:
    """Simple linear congruential generator: U in (0,1)."""
    def __init__(self, seed, a, c, m):
        self.X = seed
        self.a = a
        self.c = c
        self.m = m

    def rand(self):
        self.X = (self.a * self.X + self.c) % self.m
        return self.X / float(self.m)


def f(x):
    """
    Integrand f(x). Replace with the given function in the exam.

    Example:
        Return x*x for ∫ x^2 dx.
    """
    return math.exp(-x)  # EDIT this in exam if needed


def main():
    rng = LCG(seed, A, C, M)

    with open(output_file, "w") as f_out:
        f_out.write("# Crude Monte Carlo integration (1D)\n")
        f_out.write("# col1: N (samples)\n")
        f_out.write("# col2: I_N   (MC estimate of integral)\n")
        f_out.write("# col3: Var_I_N (numerical variance estimate of I_N)\n")
        f_out.write("# col4: error_N = |I_N - I_prev|\n")

        I_prev = None

        N_vals = []
        I_vals = []
        Var_vals = []
        Err_vals = []

        # Running statistics for Y = f(X) using Welford's algorithm
        total_N = 0
        mean_f = 0.0
        M2_f = 0.0  # sum of squared deviations

        for N_target in N_list:
            # Continue sampling up to N_target
            while total_N < N_target:
                u = rng.rand()
                x = a + (b - a) * u
                y = f(x)

                total_N += 1
                # Welford update for mean_f and variance of f(X)
                delta = y - mean_f
                mean_f += delta / float(total_N)
                delta2 = y - mean_f
                M2_f += delta * delta2

            # current sample variance of f(X)
            if total_N > 1:
                var_f = M2_f / float(total_N - 1)
            else:
                var_f = 0.0

            I_N = (b - a) * mean_f
            Var_I_N = (b - a) ** 2 * var_f / float(total_N)

            if I_prev is None:
                error = 0.0
            else:
                error = abs(I_N - I_prev)

            f_out.write(f"{total_N} {I_N} {Var_I_N} {error}\n")

            N_vals.append(total_N)
            I_vals.append(I_N)
            Var_vals.append(Var_I_N)
            Err_vals.append(error)

            I_prev = I_N

    print("Crude MC 1D data written to:", output_file)
    print("Example gnuplot commands:")
    print("  plot 'mc_crude1d.dat' using 1:2 with linespoints title 'I_N'")
    print("  plot 'mc_crude1d.dat' using 1:4 with linespoints title 'error'")

    if HAVE_MPL:
        # Plot I_N vs N
        plt.figure()
        plt.plot(N_vals, I_vals, marker='o')
        plt.xlabel("N")
        plt.ylabel("I_N")
        plt.title("Crude MC 1D: estimate vs N")

        # Plot error vs N (skip first zero-error point)
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in Err_vals]
        plt.figure()
        if len(N_vals) > 1:
            plt.semilogy(N_vals[1:], err_no_zero[1:], marker='o')
        plt.xlabel("N")
        plt.ylabel("|I_N - I_prev|")
        plt.title("Crude MC 1D: numerical error")
        plt.show()


if __name__ == "__main__":
    main()