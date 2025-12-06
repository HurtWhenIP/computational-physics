"""
Metropolis–Hastings (general proposal q(y|x)) for a 1D target distribution.

Goal:
    Generate samples X_n from a target distribution with unnormalized density
        f_target_unnorm(x)
    on a finite interval [x_min, x_max].

Algorithm (general MH):
    - At step n, current state is x.
    - Propose y ~ q(y | x) using sample_proposal(rng, x).
    - Compute acceptance probability:
          alpha = min(1,
                      [f(y) * q(x | y)] / [f(x) * q(y | x)]
                     ).

      Here:
          f(x)       = f_target_unnorm(x)
          q(y | x)   = conditional proposal density
          q(x | y)   = conditional density for the reverse move.

      Special case: if the proposal is symmetric, q(y | x) = q(x | y),
      then the ratio simplifies to:
          alpha = min(1, f(y)/f(x)),
      which is the usual "random walk Metropolis" formula.

This script:
    - Uses an LCG-based uniform RNG.
    - Allows you to define in the "TARGET & PROPOSAL" section:
          f_target_unnorm(x)
          sample_proposal(rng, x_current)   # draw y ~ q(y | x_current)
          q_cond_pdf(y, x_current)          # return q(y | x_current)
    - Runs a chain of length N_total, discards 'burn_in' points, and then:
        * Writes the kept chain to 'mh_general_chain.dat'.
        * Tracks running statistics on the kept samples:
              mean, variance, lag-1 autocorrelation, acceptance rate.
        * Writes summary evolution to 'mh_general_stats.dat'.

Numerical statistics (NO analytic true values):
    - mean_N       = sample mean of kept states
    - var_N        = sample variance (population form)
    - rho1_N       = numerical lag-1 autocorrelation estimate
    - acc_rate_N   = accepted_moves / total_moves
    - error_mean_N = |mean_N - mean_prev|

Data files:
    1) mh_general_chain.dat
        col1: n     (index of kept sample, starting at 1)
        col2: x_n   (state value)

    2) mh_general_stats.dat
        col1: n                (number of kept samples so far)
        col2: mean_n
        col3: var_n
        col4: rho1_n           (lag-1 autocorrelation estimate)
        col5: acc_rate         (overall acceptance rate up to this point)
        col6: error_mean_n     (|mean_n - mean_prev|)

How to use in the exam:
    - Step 1: In USER PARAMETERS, set:
          x_min, x_max, N_total, burn_in, initial_x.
    - Step 2: Edit f_target_unnorm(x) for the required target.
    - Step 3: Edit sample_proposal(rng, x_current) and q_cond_pdf(y, x_current)
              so that they describe your chosen proposal q(y|x).
    - Step 4: Run:
          python3 mh_general_1d.py
    - Step 5: In gnuplot:
          plot 'mh_general_chain.dat' using 1:2 with lines title 'chain'
          plot 'mh_general_stats.dat' using 1:2 with lines title 'mean_n'
          plot 'mh_general_stats.dat' using 1:6 with lines title 'error_mean'

Matplotlib (optional):
    - Time-series plot of x_n vs n.
    - Histogram of samples with target pdf overlaid (computed from
      f_target_unnorm numerically and normalized).
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
x_min = -4.0              # left boundary of (effective) support
x_max =  4.0              # right boundary of (effective) support

N_total = 5000            # total MH steps (including burn-in)
burn_in = 1000            # first 'burn_in' states discarded from stats/output

initial_x = 0.0           # starting point of chain

chain_file = "mh_general_chain.dat"
stats_file = "mh_general_stats.dat"

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
        """Return a single uniform(0,1) variate."""
        self.X = (self.a * self.X + self.c) % self.m
        return self.X / float(self.m)


# ---------- TARGET & PROPOSAL (EDIT THESE FOR YOUR PROBLEM) ----------

def f_target_unnorm(x):
    """
    Unnormalized target density f(x).

    Example: standard normal-like target with a small bump:
        f(x) ∝ exp(-x^2 / 2) * (1 + 0.5 sin(3x))
    on [x_min, x_max]. Outside this range we define f(x) = 0.
    """
    if x < x_min or x > x_max:
        return 0.0
    return math.exp(-0.5 * x * x) * (1.0 + 0.5 * math.sin(3.0 * x))


def sample_proposal(rng, x_current):
    """
    Proposal sampler: y ~ q(y | x_current).

    Example here: Gaussian random walk proposal:
        y = x_current + Normal(0, sigma^2).

    - This is a *conditional* proposal: it depends on the current state.
    - The corresponding conditional density is coded in q_cond_pdf(y, x).

    Other standard distribution examples:
    - Symmetric random walk (Gaussian centered at x_current):
        y = x_current + sigma * Normal(0, 1)
    
    - Uniform random walk on interval:
        u = rng.rand()
        y = x_current + delta * (2*u - 1)  # uniform in [-delta, +delta]

    Note:
        The proposal itself lives on the whole real line, but the target
        is only nonzero on [x_min, x_max]. If a proposal lands outside the
        target support, f_target_unnorm(y) = 0 and it will typically be
        rejected by MH.
    """
    sigma = 0.7

    # Box–Muller transform for one Normal(0,1) variate
    u1 = rng.rand()
    u2 = rng.rand()
    R = math.sqrt(-2.0 * math.log(1.0 - u1))
    theta = 2.0 * math.pi * u2
    z = R * math.cos(theta)         # standard Normal(0,1)

    step = sigma * z
    y = x_current + step
    return y


def q_cond_pdf(y, x):
    """
    Conditional proposal density q(y | x).

    This must match what sample_proposal(rng, x) is doing.

    For the Gaussian random walk example:
        y | x ~ Normal(mean = x, variance = sigma^2)

    So:
        q(y | x) = (1 / (sqrt(2π) sigma)) * exp( - (y - x)^2 / (2 sigma^2) )

    Other standard distribution examples:
    - Symmetric Gaussian random walk:
        q(y | x) = (1 / (sqrt(2π) sigma)) * exp(-(y - x)^2 / (2 sigma^2))
        Note: q(y | x) = q(x | y) (symmetric)
    
    - Asymmetric Gaussian (different widths):
        q(y | x) = Normal(mean = x + mu, std = sigma_q)
        Must compute both q(y | x) and q(x | y) with different means
    
    - Uniform random walk:
        q(y | x) = 1 / (2 * delta) for |y - x| <= delta
        q(y | x) = 0 otherwise
        (symmetric when delta is the same both ways)

    Because this Normal is symmetric in (x, y) (same sigma both ways),
    we also have:
        q(x | y) = q(y | x),
    but we still keep the *general* MH ratio using q(x|y) and q(y|x).
    """
    sigma = 0.7
    # Normal density with mean x, std dev sigma, evaluated at y
    z = (y - x) / sigma
    return math.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)


# ----------------------------------------------------------------------
# Main MH routine (general proposal q(y|x))
# ----------------------------------------------------------------------

def main():
    rng = LCG(seed, A, C, M)

    # Initialize current state and its target density value
    x_current = initial_x
    f_current = f_target_unnorm(x_current)

    # If starting point has zero target density, move to a reasonable point
    if f_current <= 0.0:
        # Simple fix: sample uniformly over [x_min, x_max] until f>0
        while True:
            u = rng.rand()
            candidate = x_min + (x_max - x_min) * u
            f_candidate = f_target_unnorm(candidate)
            if f_candidate > 0.0:
                x_current = candidate
                f_current = f_candidate
                break

    # Open files for chain and statistics
    f_chain = open(chain_file, "w")
    f_chain.write("# MH general-proposal chain (after burn-in)\n")
    f_chain.write("# col1: n (kept index)\n")
    f_chain.write("# col2: x_n\n")

    f_stats = open(stats_file, "w")
    f_stats.write("# MH general-proposal stats on kept samples\n")
    f_stats.write("# col1: n (kept count)\n")
    f_stats.write("# col2: mean_n\n")
    f_stats.write("# col3: var_n (population)\n")
    f_stats.write("# col4: rho1_n (lag-1 autocorr)\n")
    f_stats.write("# col5: acc_rate (overall acceptance)\n")
    f_stats.write("# col6: error_mean_n = |mean_n - mean_prev|\n")

    # Counters and sums for kept samples
    kept_count = 0
    sum_x = 0.0
    sum_x2 = 0.0
    sum_xx1 = 0.0      # sum over i>=1 of x_i * x_{i-1} (for lag-1 covariance)
    x_prev_kept = None

    # Acceptance statistics over ALL MH moves (including burn-in)
    total_moves = 0
    accepted_moves = 0

    # For numerical convergence of the running mean
    mean_prev = None

    # Lists used only for plotting
    kept_index_list = []
    mean_list = []
    var_list = []
    rho1_list = []
    acc_rate_list = []
    err_mean_list = []
    chain_values = []   # kept samples only

    # ===================== MH ITERATION LOOP ======================
    for step in range(N_total):
        total_moves += 1

        # ----- Propose a new point y from q(y | x_current) -----
        y = sample_proposal(rng, x_current)
        f_y = f_target_unnorm(y)

        # Compute proposal densities q(y|x_current) and q(x_current|y)
        q_y_given_x = q_cond_pdf(y, x_current)
        q_x_given_y = q_cond_pdf(x_current, y)

        # ----- Compute MH acceptance probability alpha -----
        # Careful with zeros to avoid division-by-zero / NaNs.
        if f_current <= 0.0 or q_y_given_x <= 0.0:
            # If current state has zero target density OR the proposal
            # density q(y|x_current) is zero, we set alpha = 0 (reject).
            alpha = 0.0
        else:
            if f_y <= 0.0 or q_x_given_y <= 0.0:
                # Proposed state has zero target density, or reverse
                # move has zero proposal density => ratio = 0.
                alpha = 0.0
            else:
                # General MH ratio:
                #   alpha = min(1, [f(y) q(x|y)] / [f(x) q(y|x)])
                ratio = (f_y * q_x_given_y) / (f_current * q_y_given_x)
                alpha = 1.0 if ratio >= 1.0 else ratio

        # ----- Accept / reject move -----
        u = rng.rand()
        if u < alpha:
            # Accept: move to y
            accepted_moves += 1
            x_current = y
            f_current = f_y
        # else: Reject: stay at current x_current

        # ----- Burn-in handling -----
        if step < burn_in:
            # Do not record or use the samples for statistics yet
            continue

        # ----- Record kept sample and update running statistics -----
        kept_count += 1
        x = x_current

        # Write chain value to file
        f_chain.write(f"{kept_count} {x}\n")
        chain_values.append(x)

        # Update sums for mean and variance
        sum_x += x
        sum_x2 += x * x

        # Update sum for lag-1 covariance
        if x_prev_kept is not None:
            sum_xx1 += x * x_prev_kept
        x_prev_kept = x

        # Running mean and population variance
        mean = sum_x / float(kept_count)
        var = (sum_x2 / float(kept_count)) - mean * mean

        # Lag-1 autocorrelation estimate
        if kept_count > 1 and var > 0.0:
            cov1 = (sum_xx1 / float(kept_count - 1)) - mean * mean
            rho1 = cov1 / var
        else:
            rho1 = 0.0

        # Overall acceptance rate up to now (over all moves)
        acc_rate = accepted_moves / float(total_moves)

        # Numerical error for mean: difference from previous mean
        if mean_prev is None:
            err_mean = 0.0
        else:
            err_mean = abs(mean - mean_prev)

        # Write stats line
        f_stats.write(f"{kept_count} {mean} {var} {rho1} {acc_rate} {err_mean}\n")

        # Store for plotting
        kept_index_list.append(kept_count)
        mean_list.append(mean)
        var_list.append(var)
        rho1_list.append(rho1)
        acc_rate_list.append(acc_rate)
        err_mean_list.append(err_mean)

        mean_prev = mean

    # ===================== END OF MH LOOP =========================

    f_chain.close()
    f_stats.close()

    print("General MH chain written to:", chain_file)
    print("Stats written to:", stats_file)
    print("Final acceptance rate:", accepted_moves / float(total_moves))

    # ------------------------------------------------------------------
    # Matplotlib sanity plots (optional)
    # ------------------------------------------------------------------
    if HAVE_MPL and kept_count > 0:
        # ---- Time-series plot of the kept chain ----
        plt.figure()
        plt.plot(kept_index_list, chain_values, linestyle='-', marker='.')
        plt.xlabel("n (kept index)")
        plt.ylabel("x_n")
        plt.title("MH general proposal: chain trace")

        # ---- Histogram + target pdf overlay ----
        plt.figure()
        n_hist, bins, _ = plt.hist(chain_values, bins=30, edgecolor="black")

        # Numerically normalize f_target_unnorm on [x_min, x_max]
        ngrid = 2000
        xs_grid = [x_min + (x_max - x_min) * i / float(ngrid)
                   for i in range(ngrid + 1)]
        vals = [f_target_unnorm(x) for x in xs_grid]
        Z = sum(vals) * ((x_max - x_min) / float(ngrid))  # approximate integral

        if Z > 0.0:
            pdf = [v / Z for v in vals]
        else:
            pdf = [0.0 for _ in vals]

        # Scale pdf to match histogram counts
        bin_width = bins[1] - bins[0]
        scale = len(chain_values) * bin_width
        pdf_scaled = [p * scale for p in pdf]

        plt.plot(xs_grid, pdf_scaled, "r-", linewidth=2, label="target pdf")
        plt.xlabel("x")
        plt.ylabel("Count")
        plt.title("MH general proposal: histogram + target pdf")
        plt.legend()

        # ---- Numerical error of mean vs n (log scale) ----
        if kept_count > 1:
            eps = 1e-16
            err_no_zero = [e if e > 0.0 else eps for e in err_mean_list]
            if len(kept_index_list) > 2:
                plt.figure()
                plt.semilogy(kept_index_list[2:], err_no_zero[2:], marker='o')
                plt.xlabel("n (kept)")
                plt.ylabel("|mean_n - mean_prev|")
                plt.title("MH general proposal: numerical convergence of mean")

        plt.show()


if __name__ == "__main__":
    main()