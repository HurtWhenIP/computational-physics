"""
Inverse-transform sampling for an exponential distribution.

Target distribution:
    X ~ Exp(lambda),  x >= 0
    CDF:  F(x) = 1 - exp(-lambda x)
    Inverse CDF:  F^{-1}(u) = - (1/lambda) * ln(1 - u)

What this script does:
- Uses an LCG to generate U ~ Uniform(0,1).
- Transforms to exponential samples:
      X = - (1/lambda) * ln(1 - U)
- Tracks running estimates of:
      mean_N    = (1/N) sum X_i
      var_N     = sample variance
- Numerical convergence:
      error_N   = |mean_N - mean_{N_prev}|
  (no analytic mean used in code for error).
- Writes 'inv_exp.dat'.
- Also stores all samples (up to max N) and, if matplotlib is available,
  plots a histogram of the obtained exponential distribution.

How to use in the exam:
- Edit USER PARAMETERS:
      lam, N_list.
- Run:
      python3 mc_inverse_exponential.py
- Gnuplot example:
      plot 'inv_exp.dat' using 1:3 with linespoints title 'mean_N'
      plot 'inv_exp.dat' using 1:5 with linespoints title 'error'

Data file columns:
    col1: N
    col2: last_sample_X
    col3: mean_N
    col4: var_N
    col5: error_N = |mean_N - mean_prev|
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
seed = 12345
a    = 1103515245
c    = 12345
m    = 2**31 - 1

lam  = 1.0                   # lambda > 0
N_list = [100, 500, 1000, 5000]
output_file = "inv_exp.dat"
# ===============================================================


class LCG:
    def __init__(self, seed, a, c, m):
        self.X = seed
        self.a = a
        self.c = c
        self.m = m

    def rand(self):
        self.X = (self.a * self.X + self.c) % self.m
        return self.X / float(self.m)


def main():
    rng = LCG(seed, a, c, m)

    with open(output_file, "w") as f:
        f.write("# Inverse-transform exponential sampling\n")
        f.write("# col1: N\n")
        f.write("# col2: last_sample_X\n")
        f.write("# col3: running mean_N\n")
        f.write("# col4: running var_N (sample)\n")
        f.write("# col5: error_N = |mean_N - mean_prev|\n")

        N_vals = []
        mean_vals = []
        var_vals = []
        err_vals = []

        prev_mean = None

        # Welford's algorithm for stable online variance
        total_N = 0
        mean = 0.0
        M2 = 0.0  # sum of squares of differences

        # Store samples for distribution plot (up to max N)
        samples = []

        for N_target in N_list:
            # Generate up to N_target (continuing from previous sample count)
            while total_N < N_target:
                u = rng.rand()
                # avoid log(0)
                if u == 0.0:
                    u = 1e-16
                x = -math.log(1.0 - u) / lam

                total_N += 1
                samples.append(x)

                # Welford update
                delta = x - mean
                mean += delta / float(total_N)
                delta2 = x - mean
                M2 += delta * delta2

            if total_N > 1:
                var = M2 / float(total_N - 1)
            else:
                var = 0.0

            if prev_mean is None:
                error = 0.0
            else:
                error = abs(mean - prev_mean)

            f.write(f"{total_N} {x} {mean} {var} {error}\n")

            N_vals.append(total_N)
            mean_vals.append(mean)
            var_vals.append(var)
            err_vals.append(error)

            prev_mean = mean

    print("Inverse-transform exponential data written to:", output_file)
    print("Example gnuplot:")
    print("  plot 'inv_exp.dat' using 1:3 with linespoints title 'mean_N'")
    print("  plot 'inv_exp.dat' using 1:5 with linespoints title 'error'")

    if HAVE_MPL:
        # Plot distribution (histogram of samples)
                # --- Histogram of samples ---
        plt.figure()
        n, bins, _ = plt.hist(samples, bins=30, edgecolor="black")
        plt.xlabel("x")
        plt.ylabel("Count")
        plt.title("Exponential distribution (inverse transform)")

        # ===== Overlay target exponential pdf with correct scaling =====
        bin_width = bins[1] - bins[0]
        xmin = bins[0]
        xmax = bins[-1]
        xr = [xmin + (xmax - xmin) * i / 500.0 for i in range(501)]
        pdf = [lam * math.exp(-lam * x) for x in xr]   # exponential pdf
        scale = len(samples) * bin_width
        pdf_scaled = [p * scale for p in pdf]
        plt.plot(xr, pdf_scaled, "r-", linewidth=2, label="target pdf")
        plt.legend()

        # --- Convergence plot: skip first zero-error point ---
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]
        plt.figure()
        plt.semilogy(N_vals[1:], err_no_zero[1:], marker="o")
        plt.xlabel("N")
        plt.ylabel("|mean_N - mean_prev|")
        plt.title("Exponential: numerical convergence of mean")

        plt.show()


if __name__ == "__main__":
    main()