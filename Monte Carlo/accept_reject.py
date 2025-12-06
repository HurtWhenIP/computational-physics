"""
Generic accept–reject sampling template.

We want to sample from a target PDF f(x) known up to a constant, using
a simpler proposal PDF g(x) and a constant M such that:
    f(x) <= M * g(x)  for all x in the support.

Algorithm:
1) Sample Y ~ g.
2) Sample U ~ Uniform(0,1).
3) Accept X = Y if:
       U <= f(Y) / (M * g(Y)),
   else reject and repeat.

Example in this script:
- Target: f(x) proportional to exp(-x^2/2) on [x_min, x_max].
- Proposal: Uniform(x_min, x_max).

You can change f_target_unnorm(x), sample_proposal(), g_pdf(), x_min, x_max
to match ANY target and proposal allowed in the exam.

What this script does:
- Implements the accept–reject sampler using an LCG.
- Tracks:
      acceptance rate,
      running mean of X,
      running variance of X,
      numerical convergence error = |mean_N - mean_prev|.
- Writes 'accept_reject.dat'.
- Also stores all accepted samples and, if matplotlib is available,
  plots:
      * histogram of accepted samples
      * red curve of the TARGET pdf built directly from f_target_unnorm(x)
        (numerically normalized + scaled to match histogram counts).

How to use in the exam:
- Edit USER PARAMETERS (N_target, support, and definitions of f_target_unnorm,
  sample_proposal and g_pdf).
- Run:
      python3 mc_accept_reject.py
- Gnuplot:
      plot 'accept_reject.dat' using 1:4 with linespoints title 'mean_N'
      plot 'accept_reject.dat' using 1:6 with linespoints title 'error'

Data file columns:
    col1: N_accepted
    col2: last_sample_X
    col3: acceptance_rate
    col4: mean_N
    col5: var_N
    col6: error_N
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

N_target = 2000             # how many accepted samples to generate
x_min    = -3.0             # left end of support
x_max    =  3.0             # right end of support
output_file = "accept_reject.dat"
# ===============================================================


class LCG:
    """
    Simple linear congruential generator.

    Usage:
        rng = LCG(seed, a, c, m)
        u = rng.rand()   # uniform in (0,1)
    """
    def __init__(self, seed, a, c, m):
        self.X = seed
        self.a = a
        self.c = c
        self.m = m

    def rand(self):
        self.X = (self.a * self.X + self.c) % self.m
        return self.X / float(self.m)


# ---------- TARGET AND PROPOSAL: EDIT THESE FOR YOUR PROBLEM ----------

def f_target_unnorm(x):
    """
    Unnormalized target density f(x) (non-negative, not necessarily integrate to 1).

    Example here: truncated normal-like on [x_min, x_max]:
        f(x) ∝ exp(-x^2 / 2).

    In the exam, REPLACE this with the given f(x) (up to constant).
    Make sure it returns 0 outside [x_min, x_max] if the density support
    is restricted to this interval.
    """
    if x < x_min or x > x_max:
        return 0.0
    return math.exp(-0.5 * x * x) * x*x * math.atan(math.exp(math.sin(3*x+69)))


def sample_proposal(rng):
    """
    Sample from proposal g(x).

    Example here: Uniform(x_min, x_max).
    Modify if you use another proposal distribution.
    """
    u = rng.rand()
    return x_min + (x_max - x_min) * u


def g_pdf(x):
    """
    Proposal PDF g(x).

    Example here: Uniform(x_min, x_max):
        g(x) = 1 / (x_max - x_min) inside the interval,
               0 outside.
    """
    if x < x_min or x > x_max:
        return 0.0
    return 1.0 / (x_max - x_min)


# ----------------------------------------------------------------------
# Helper to choose M numerically
# ----------------------------------------------------------------------

def find_M_grid():
    """
    Simple numerical search to find a safe constant M such that:
        f_target_unnorm(x) <= M * g_pdf(x)
    on a grid in [x_min, x_max].

    In the exam, you can:
      - either use this helper, or
      - hard-code M if an analytic bound is obvious.
    """
    max_ratio = 0.0
    n_grid = 1000
    for i in range(n_grid + 1):
        x = x_min + (x_max - x_min) * (i / float(n_grid))
        fx = f_target_unnorm(x)
        gx = g_pdf(x)
        if gx > 0.0:
            r = fx / gx
            if r > max_ratio:
                max_ratio = r
    # Safety factor
    return 1.1 * max_ratio


# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------

def main():
    rng = LCG(seed, a, c, m)

    M = find_M_grid()
    print("Using M =", M, "for accept–reject.")

    with open(output_file, "w") as f:
        f.write("# Accept–reject sampling data\n")
        f.write("# col1: N_accepted\n")
        f.write("# col2: last_X\n")
        f.write("# col3: acceptance_rate\n")
        f.write("# col4: mean_N\n")
        f.write("# col5: var_N\n")
        f.write("# col6: error_N = |mean_N - mean_prev|\n")

        N_accepted = 0     # number of accepted samples
        N_total    = 0     # total proposals generated
        mean = 0.0
        M2   = 0.0
        prev_mean = None

        N_vals = []
        mean_vals = []
        var_vals = []
        err_vals = []
        acc_vals = []
        accepted_samples = []

        last_x = 0.0

        # -------------- Accept–reject loop --------------
        while N_accepted < N_target:
            y = sample_proposal(rng)
            u = rng.rand()

            fy = f_target_unnorm(y)
            gy = g_pdf(y)
            if gy <= 0.0:
                # proposal has zero density here, skip this point
                continue

            accept_prob = fy / (M * gy)

            if u <= accept_prob:
                # accepted
                N_accepted += 1
                last_x = y
                accepted_samples.append(y)

                # Online mean/variance (Welford)
                delta = y - mean
                mean += delta / float(N_accepted)
                delta2 = y - mean
                M2 += delta * delta2

            N_total += 1

            # Running variance
            if N_accepted > 1:
                var = M2 / float(N_accepted - 1)
            else:
                var = 0.0

            # Numerical convergence of mean (no analytic mean used)
            if prev_mean is None:
                error = 0.0
            else:
                error = abs(mean - prev_mean)

            acc_rate = N_accepted / float(N_total)

            f.write(f"{N_accepted} {last_x} {acc_rate} {mean} {var} {error}\n")

            N_vals.append(N_accepted)
            mean_vals.append(mean)
            var_vals.append(var)
            err_vals.append(error)
            acc_vals.append(acc_rate)

            prev_mean = mean

    print("Accept–reject data written to:", output_file)
    if N_vals:
        print("Final acceptance rate:", acc_vals[-1])

    # ------------------------------------------------------------------
    # Matplotlib sanity plots (if available)
    # ------------------------------------------------------------------
    if HAVE_MPL and accepted_samples:
        # --- Histogram of accepted samples ---
        plt.figure()
        n, bins, _ = plt.hist(accepted_samples, bins=30, edgecolor="black")
        plt.xlabel("x")
        plt.ylabel("Count")
        plt.title("Accept–reject: sampled distribution")

        # ===== Overlay target pdf BUILT FROM f_target_unnorm(x) =====
        # 1) Numerically approximate normalization constant:
        #       Z ≈ ∫ f_target_unnorm(x) dx over [x_min, x_max]
        ngrid = N_target
        xs_grid = [x_min + (x_max - x_min) * i / ngrid for i in range(ngrid + 1)]
        vals = [f_target_unnorm(x) for x in xs_grid]
        Z = sum(vals) * ((x_max - x_min) / ngrid)   # approximate integral

        # 2) Normalized pdf(x) = f_target_unnorm(x) / Z
        xr = xs_grid
        pdf = [v / Z for v in vals]

        # 3) Scale pdf so that its area roughly matches histogram counts:
        #    expected bin count ≈ N * pdf(x) * bin_width
        bin_width = bins[1] - bins[0]
        scale = len(accepted_samples) * bin_width
        pdf_scaled = [p * scale for p in pdf]

        plt.plot(xr, pdf_scaled, "r-", linewidth=2, label="target pdf from f(x)")
        plt.legend()

        # --- Convergence plot: skip first zero-error point ---
        if len(N_vals) > 1:
            eps = 1e-16
            err_no_zero = [e if e > 0.0 else eps for e in err_vals]
            plt.figure()
            plt.semilogy(N_vals[1:], err_no_zero[1:], marker="o")
            plt.xlabel("N_accepted")
            plt.ylabel("|mean_N - mean_prev|")
            plt.title("Accept–reject: numerical convergence of mean")

        plt.show()


if __name__ == "__main__":
    main()