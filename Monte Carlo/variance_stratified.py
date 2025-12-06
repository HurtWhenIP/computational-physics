"""
Stratified sampling for 1D Monte Carlo integration.

Goal:
    Estimate I = ∫_a^b f(x) dx.

Stratified sampling:
    - Split [a,b] into S equal strata:
          [x_0, x_1], [x_1, x_2], ..., [x_{S-1}, x_S]
      where x_j = a + j * (b-a)/S.
    - In each stratum j, sample n_per_stratum points uniformly and take
      the average of f(x).
    - Stratum width: Δ = (b-a)/S
    - Estimator:
          I_strat = Σ_{j=1..S} Δ * mean_j

Crude MC (comparison):
    - Use N_total = S * n_per_stratum samples uniformly over [a,b].
    - Estimator:
          I_crude = (b-a) * mean_over_all_samples.

This script:
    - For each S in S_list:
        * Uses the same total N = S * n_per_stratum.
        * Computes I_crude, I_strat, and numerical variance estimates for each.
    - Numerical errors (within a single run over S_list):
        * error_crude(S) = |I_crude(S) - I_crude(prev S)|
        * error_strat(S) = |I_strat(S) - I_strat(prev S)|

Outputs:
    - 'mc_stratified.dat'

        col1: S                    (number of strata)
        col2: N_total              (samples = S * n_per_stratum)
        col3: I_crude              (crude MC estimate)
        col4: Var_crude            (variance estimate)
        col5: I_strat              (stratified estimate)
        col6: Var_strat            (variance estimate based on stratum means)
        col7: error_crude
        col8: error_strat

How to use in the exam:
    - Edit f(x), a, b, S_list, n_per_stratum.
    - Run:
        python3 mc_variance_stratified.py
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
S_list = [2, 4, 8, 16]       # number of strata
n_per_stratum = 100          # samples in each stratum
output_file = "mc_stratified.dat"

seed = 12345
A    = 1103515245
C    = 12345
M    = 2**31 - 1
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


def f(x):
    """
    Integrand f(x).

    Example: f(x) = sin(pi x).
    """
    return math.sin(math.pi * x)


def main():
    rng = LCG(seed, A, C, M)

    with open(output_file, "w") as f_out:
        f_out.write("# Stratified sampling vs crude MC\n")
        f_out.write("# col1: S (number of strata)\n")
        f_out.write("# col2: N_total = S * n_per_stratum\n")
        f_out.write("# col3: I_crude\n")
        f_out.write("# col4: Var_crude\n")
        f_out.write("# col5: I_strat\n")
        f_out.write("# col6: Var_strat\n")
        f_out.write("# col7: error_crude\n")
        f_out.write("# col8: error_strat\n")

        I_crude_prev = None
        I_strat_prev = None

        S_values = []
        I_crude_values = []
        Var_crude_values = []
        I_strat_values = []
        Var_strat_values = []
        Err_crude_values = []
        Err_strat_values = []

        for S in S_list:
            width = (b - a) / float(S)
            N_total = S * n_per_stratum

            # ---- Crude MC over all samples ----
            sum_f = 0.0
            sum_f2 = 0.0

            for _ in range(N_total):
                u = rng.rand()
                x = a + (b - a) * u
                y = f(x)
                sum_f += y
                sum_f2 += y * y

            mean_f = sum_f / float(N_total)
            if N_total > 1:
                var_f = (sum_f2 - N_total * mean_f * mean_f) / float(N_total - 1)
            else:
                var_f = 0.0

            I_crude = (b - a) * mean_f
            Var_crude = (b - a) ** 2 * var_f / float(N_total)

            # ---- Stratified sampling ----
            stratum_means = []

            for j in range(S):
                left = a + j * width
                right = left + width

                # average f in this stratum
                sum_str = 0.0
                for _ in range(n_per_stratum):
                    u = rng.rand()
                    x = left + (right - left) * u
                    y = f(x)
                    sum_str += y
                mean_str = sum_str / float(n_per_stratum)
                stratum_means.append(mean_str)

            # stratified estimate
            I_strat = 0.0
            for m_j in stratum_means:
                I_strat += width * m_j

            # variance estimate based on stratum means:
            # treat the S stratum estimators as a small sample.
            if S > 1:
                mean_est = I_strat  # mean of estimators (single run)
                # approximate var via sample var of (width * mean_j)
                vals = [width * m_j for m_j in stratum_means]
                avg = sum(vals) / float(S)
                sum_sq = 0.0
                for v in vals:
                    sum_sq += (v - avg) ** 2
                Var_strat = sum_sq / float(S - 1) / float(S)
            else:
                Var_strat = 0.0

            if I_crude_prev is None:
                err_crude = 0.0
            else:
                err_crude = abs(I_crude - I_crude_prev)

            if I_strat_prev is None:
                err_strat = 0.0
            else:
                err_strat = abs(I_strat - I_strat_prev)

            f_out.write(f"{S} {N_total} {I_crude} {Var_crude} "
                        f"{I_strat} {Var_strat} {err_crude} {err_strat}\n")

            S_values.append(S)
            I_crude_values.append(I_crude)
            Var_crude_values.append(Var_crude)
            I_strat_values.append(I_strat)
            Var_strat_values.append(Var_strat)
            Err_crude_values.append(err_crude)
            Err_strat_values.append(err_strat)

            I_crude_prev = I_crude
            I_strat_prev = I_strat

    print("Stratified sampling data written to:", output_file)

    if HAVE_MPL:
        plt.figure()
        plt.plot(S_values, Var_crude_values, marker='o', label="Var(crude)")
        plt.plot(S_values, Var_strat_values, marker='s', label="Var(stratified)")
        plt.xlabel("S (strata)")
        plt.ylabel("Var[estimator]")
        plt.title("Stratified vs crude: variance")
        plt.legend()

        eps = 1e-16
        err_c_no_zero = [e if e > 0.0 else eps for e in Err_crude_values]
        err_s_no_zero = [e if e > 0.0 else eps for e in Err_strat_values]
        if len(S_values) > 1:
            plt.figure()
            plt.semilogy(S_values[1:], err_c_no_zero[1:], marker='o',
                         label="crude error")
            plt.semilogy(S_values[1:], err_s_no_zero[1:], marker='s',
                         label="stratified error")
            plt.xlabel("S (strata)")
            plt.ylabel("|I(S) - I(prev S)|")
            plt.title("Stratified vs crude: numerical error")
            plt.legend()
        plt.show()


if __name__ == "__main__":
    main()