"""
Variance reduction using antithetic variables (1D integral).

Goal:
    Estimate I = ∫_a^b f(x) dx.

Crude MC:
    - Sample U ~ Uniform(0,1), set X = a + (b - a) U.
    - Estimator per sample:
        Y_crude = f(X).

Antithetic MC:
    - Use pairs (U, 1 - U).
    - Within each pair:
        X1 = a + (b - a) U
        X2 = a + (b - a) (1 - U)
        Y1 = f(X1), Y2 = f(X2)
      Pair estimator:
        Y_pair = (Y1 + Y2) / 2
    - Overall integral estimator:
        I_anti = (b - a) * mean_of(Y_pair_over_pairs).

This script:
    - Uses LCG to generate U.
    - In each of N_pairs pairs:
        * Computes crude sample: Y_crude = f(X1) only.
        * Computes antithetic pair estimator Y_pair.
    - Tracks running means and variances for:
        * crude estimator values
        * antithetic pair estimator values
    - Numerical "error" metrics:
        * diff between successive I_crude(N) and I_anti(N).

Outputs:
    - 'mc_antithetic.dat'
        col1: N_pairs
        col2: I_crude_N
        col3: Var_crude_N
        col4: I_anti_N
        col5: Var_anti_N
        col6: error_crude_N = |I_crude_N - I_crude_prev|
        col7: error_anti_N  = |I_anti_N  - I_anti_prev|

How to use in the exam:
    - Edit a, b, N_pairs_list, and f(x).
    - Run:
        python3 mc_variance_antithetic.py
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
N_pairs_list = [50, 200, 1000, 5000]
output_file = "mc_antithetic.dat"

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

    Example: f(x) = exp(-x^2).
    """
    return math.exp(-x * x)


def main():
    rng = LCG(seed, A, C, M)

    with open(output_file, "w") as f_out:
        f_out.write("# Antithetic variables vs crude MC\n")
        f_out.write("# col1: N_pairs\n")
        f_out.write("# col2: I_crude_N\n")
        f_out.write("# col3: Var_crude_N\n")
        f_out.write("# col4: I_anti_N\n")
        f_out.write("# col5: Var_anti_N\n")
        f_out.write("# col6: error_crude_N = |I_crude_N - I_crude_prev|\n")
        f_out.write("# col7: error_anti_N  = |I_anti_N  - I_anti_prev|\n")

        # Running stats for crude: values = Y_crude = f(X1)
        n_pairs = 0
        mean_crude = 0.0
        M2_crude = 0.0

        # Running stats for antithetic: values = Y_pair = (Y1 + Y2)/2
        mean_anti = 0.0
        M2_anti = 0.0

        I_crude_prev = None
        I_anti_prev  = None

        N_list = []
        I_crude_list = []
        Var_crude_list = []
        I_anti_list = []
        Var_anti_list = []
        Err_crude_list = []
        Err_anti_list = []

        for N_target in N_pairs_list:
            while n_pairs < N_target:
                u = rng.rand()
                x1 = a + (b - a) * u
                x2 = a + (b - a) * (1.0 - u)

                y1 = f(x1)
                y2 = f(x2)
                y_crude = y1           # crude uses only first
                y_pair = 0.5 * (y1 + y2)

                n_pairs += 1

                # update crude stats
                delta_c = y_crude - mean_crude
                mean_crude += delta_c / float(n_pairs)
                delta2_c = y_crude - mean_crude
                M2_crude += delta_c * delta2_c

                # update antithetic stats
                delta_a = y_pair - mean_anti
                mean_anti += delta_a / float(n_pairs)
                delta2_a = y_pair - mean_anti
                M2_anti += delta_a * delta2_a

            if n_pairs > 1:
                var_crude_val = M2_crude / float(n_pairs - 1)
                var_anti_val = M2_anti / float(n_pairs - 1)
            else:
                var_crude_val = 0.0
                var_anti_val = 0.0

            I_crude = (b - a) * mean_crude
            Var_crude = (b - a) ** 2 * var_crude_val / float(n_pairs)

            I_anti = (b - a) * mean_anti
            Var_anti = (b - a) ** 2 * var_anti_val / float(n_pairs)

            if I_crude_prev is None:
                err_crude = 0.0
            else:
                err_crude = abs(I_crude - I_crude_prev)

            if I_anti_prev is None:
                err_anti = 0.0
            else:
                err_anti = abs(I_anti - I_anti_prev)

            f_out.write(f"{n_pairs} {I_crude} {Var_crude} "
                        f"{I_anti} {Var_anti} {err_crude} {err_anti}\n")

            N_list.append(n_pairs)
            I_crude_list.append(I_crude)
            Var_crude_list.append(Var_crude)
            I_anti_list.append(I_anti)
            Var_anti_list.append(Var_anti)
            Err_crude_list.append(err_crude)
            Err_anti_list.append(err_anti)

            I_crude_prev = I_crude
            I_anti_prev  = I_anti

    print("Antithetic variables data written to:", output_file)

    if HAVE_MPL:
        plt.figure()
        plt.plot(N_list, Var_crude_list, marker='o', label="Var(crude)")
        plt.plot(N_list, Var_anti_list, marker='s', label="Var(antithetic)")
        plt.xlabel("N_pairs")
        plt.ylabel("Var[estimator]")
        plt.title("Antithetic vs crude: variance estimates")
        plt.legend()

        eps = 1e-16
        err_c_no_zero = [e if e > 0.0 else eps for e in Err_crude_list]
        err_a_no_zero = [e if e > 0.0 else eps for e in Err_anti_list]
        if len(N_list) > 1:
            plt.figure()
            plt.semilogy(N_list[1:], err_c_no_zero[1:], marker='o',
                         label="crude error")
            plt.semilogy(N_list[1:], err_a_no_zero[1:], marker='s',
                         label="antithetic error")
            plt.xlabel("N_pairs")
            plt.ylabel("|I_N - I_prev|")
            plt.title("Antithetic vs crude: numerical error")
            plt.legend()
        plt.show()


if __name__ == "__main__":
    main()