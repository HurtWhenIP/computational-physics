"""
Importance sampling for 1D integrals (exam template).

Goal:
    Approximate integral
        I = ∫_a^b f(x) dx
    using a proposal PDF q(x) (possibly on a larger domain, e.g. R).

Theory:
    Write:
        I = ∫_a^b f(x) dx
          = ∫_{R} [f(x) * 1_{[a,b]}(x) / q(x)] q(x) dx
      If X ~ q, then the estimator is:
        I_N = (1/N) Σ w_i  where w_i = f(X_i) * 1_{[a,b]}(X_i) / q(X_i).

This script:
    - Uses an LCG to sample X_i from q(x) via sample_q(rng).
    - Computes weights w_i = f(X_i) * 1_{[a,b]}(X_i) / q(X_i).
    - Tracks running stats:
         mean_w(N)  = mean of weights
         var_w(N)   = sample variance of weights
         I_IS_N     = mean_w(N)
         Var[I_IS_N] ≈ var_w(N) / N
    - Compares to crude Monte Carlo with Uniform(a,b):
         I_crude_N = (b - a) * mean(f(U_i)),  U_i ~ Uniform(a,b)
    - Numerical errors (no true value):
         error_IS_N     = |I_IS_N - I_IS_prev|
         error_crude_N  = |I_crude_N - I_crude_prev|

Outputs:
    - 'mc_importance.dat'

        col1: N
        col2: I_IS_N        (importance sampling estimate)
        col3: Var_IS_N      (variance estimate of IS estimator)
        col4: error_IS_N
        col5: I_crude_N     (crude MC estimate with uniform)
        col6: Var_crude_N
        col7: error_crude_N

How to use in the exam:
    - ONLY edit:
        * a, b, N_list
        * f(x)
        * sample_q(rng)
        * q_pdf(x)
    - Everything else can be left as-is.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========

# Integration limits
a = 0.0
b = 100000.0

# Sample sizes at which to record statistics
N_list = [100, 500, 1000, 5000, 10000]

# Output file
output_file = "mc_importance.dat"

# LCG params (do not really need to change these)
seed = 12345
A    = 1103515245
C    = 12345
M    = 2**31 - 1

# ---------------------------------------------------------------
# Target integrand f(x) on [a,b].
# Only this function needs to be changed for a different problem.
# Example: f(x) = x^2 * exp(-x)
# ---------------------------------------------------------------

def f(x):
    return math.exp(-2*x)  # example integrand


# ---------------------------------------------------------------
# Proposal distribution q(x) (can be defined on R).
#
# You must keep sample_q and q_pdf consistent:
#   - sample_q(rng) generates X ~ q
#   - q_pdf(x) returns q(x) (the PDF at x)
#
# You are free to choose any q (normal, exponential, etc.).
# The code automatically restricts the integral to [a,b] via
# an indicator 1_{[a,b]}(x) inside the weight.
# ---------------------------------------------------------------

def sample_q(rng):
    """
    Sample from proposal q(x).

    Example here: standard normal N(0,1) using Box-Muller.
    You can change this to any 1D distribution, but make sure
    q_pdf(x) matches!
    """
    u1 = rng.rand()
    u2 = rng.rand()
    R = math.sqrt(-2.0 * math.log(1.0 - u1))
    theta = 2.0 * math.pi * u2
    return R * math.cos(theta)  # N(0,1)


def q_pdf(x):
    """
    Proposal PDF q(x) corresponding to sample_q.

    For the standard normal:
        q(x) = (1/sqrt(2π)) * exp(-x^2 / 2)
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ===============================================================
#                  DO NOT EDIT BELOW THIS LINE
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
    # Separate RNGs for IS and crude MC
    rng_IS = LCG(seed, A, C, M)
    rng_crude = LCG(seed + 777, A, C, M)

    with open(output_file, "w") as f_out:
        f_out.write("# Importance sampling vs crude MC (1D)\n")
        f_out.write("# col1: N\n")
        f_out.write("# col2: I_IS_N       (importance sampling)\n")
        f_out.write("# col3: Var_IS_N\n")
        f_out.write("# col4: error_IS_N = |I_IS_N - I_IS_prev|\n")
        f_out.write("# col5: I_crude_N    (uniform crude MC)\n")
        f_out.write("# col6: Var_crude_N\n")
        f_out.write("# col7: error_crude_N\n")

        # Running stats for IS weights w_i = f(X_i)*1_[a,b]/q(X_i)
        total_N_IS = 0
        mean_w = 0.0
        M2_w = 0.0

        # Running stats for crude MC: Y = f(U), U ~ Uniform(a,b)
        total_N_crude = 0
        mean_f_crude = 0.0
        M2_f_crude = 0.0

        I_IS_prev = None
        I_crude_prev = None

        # For plotting
        N_vals = []
        I_IS_vals = []
        Var_IS_vals = []
        I_crude_vals = []
        Var_crude_vals = []
        Err_IS_vals = []
        Err_crude_vals = []

        for N_target in N_list:
            while total_N_IS < N_target:
                # ---------- Importance sampling ----------
                x_IS = sample_q(rng_IS)
                qx = q_pdf(x_IS)

                if qx > 0.0 and (a <= x_IS <= b):
                    w = f(x_IS) / qx
                else:
                    # outside [a,b] or qx == 0 => indicator is 0
                    w = 0.0

                total_N_IS += 1
                delta = w - mean_w
                mean_w += delta / float(total_N_IS)
                delta2 = w - mean_w
                M2_w += delta * delta2

                # ---------- Crude MC (Uniform on [a,b]) ----------
                u = rng_crude.rand()
                x_crude = a + (b - a) * u
                y_crude = f(x_crude)

                total_N_crude += 1
                d_crude = y_crude - mean_f_crude
                mean_f_crude += d_crude / float(total_N_crude)
                d2_crude = y_crude - mean_f_crude
                M2_f_crude += d_crude * d2_crude

            # Sample variances
            if total_N_IS > 1:
                var_w = M2_w / float(total_N_IS - 1)
            else:
                var_w = 0.0

            if total_N_crude > 1:
                var_f_crude = M2_f_crude / float(total_N_crude - 1)
            else:
                var_f_crude = 0.0

            # Estimators
            I_IS = mean_w
            Var_IS = var_w / float(total_N_IS)

            I_crude = (b - a) * mean_f_crude
            Var_crude = (b - a) ** 2 * var_f_crude / float(total_N_crude)

            # Numerical errors (successive differences only)
            if I_IS_prev is None:
                err_IS = 0.0
            else:
                err_IS = abs(I_IS - I_IS_prev)

            if I_crude_prev is None:
                err_crude = 0.0
            else:
                err_crude = abs(I_crude - I_crude_prev)

            f_out.write(f"{total_N_IS} {I_IS} {Var_IS} {err_IS} "
                        f"{I_crude} {Var_crude} {err_crude}\n")

            # Store for plotting
            N_vals.append(total_N_IS)
            I_IS_vals.append(I_IS)
            Var_IS_vals.append(Var_IS)
            I_crude_vals.append(I_crude)
            Var_crude_vals.append(Var_crude)
            Err_IS_vals.append(err_IS)
            Err_crude_vals.append(err_crude)

            I_IS_prev = I_IS
            I_crude_prev = I_crude

    print("Importance sampling data written to:", output_file)

    # Optional plots (only if matplotlib is available)
    if HAVE_MPL:
        # Estimator vs N
        plt.figure()
        plt.plot(N_vals, I_IS_vals, marker='o', label="IS")
        plt.xlabel("N")
        plt.ylabel("Estimator")
        plt.title("Importance sampling")
        plt.legend()

        # Numerical error vs N (log scale), skipping first (zero error)
        eps = 1e-16
        err_IS_no_zero = [e if e > 0.0 else eps for e in Err_IS_vals]
        err_crude_no_zero = [e if e > 0.0 else eps for e in Err_crude_vals]
        if len(N_vals) > 1:
            plt.figure()
            plt.semilogy(N_vals[1:], err_IS_no_zero[1:], marker='o', label="IS")
            plt.semilogy(N_vals[1:], err_crude_no_zero[1:], marker='s', label="crude")
            plt.xlabel("N")
            plt.ylabel("|I_N - I_prev|")
            plt.title("Numerical error: IS vs crude")
            plt.legend()

        plt.show()


if __name__ == "__main__":
    main()