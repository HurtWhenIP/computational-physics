"""
Variance reduction via control variates (1D integral).

Setting:
    - We want I = E[f(X)] where X ~ Uniform(a,b).
      Then I = (1/(b-a)) ∫_a^b f(x) dx.

Control variate:
    - Choose a function h(x) with a known expectation under Uniform(a,b):
          E[h(X)] = mu_h   (known analytically).
    - For each sample X_i, define:
          Z_i = f(X_i) - beta * (h(X_i) - mu_h)
      The control-variate estimator is:
          I_cv = mean(Z_i).

Notes:
    - beta is a user-chosen parameter (can be tuned).
    - This script keeps beta as a USER PARAMETER so you can override it.

Error & variance:
    - We track numerically:
          mean_f(N), var_f(N)   for crude estimator.
          mean_Z(N), var_Z(N)   for control variate estimator.
    - Numerical "errors":
          error_crude_N = |I_crude_N - I_crude_prev|
          error_cv_N    = |I_cv_N    - I_cv_prev|

Outputs:
    - 'mc_control_variate.dat'

        col1: N
        col2: I_crude_N
        col3: Var_crude_N
        col4: I_cv_N
        col5: Var_cv_N
        col6: error_crude_N
        col7: error_cv_N

How to use in the exam:
    - Choose f(x), the control h(x), and its known E[h].
    - Set:
          a, b, N_list, mu_h, beta_cv.
    - Run:
          python3 mc_variance_control_variate.py
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
output_file = "mc_control_variate.dat"

# Control variate parameters:
# Example: h(x) = x, X ~ Uniform(0,1) => E[h] = 1/2.
mu_h = 0.5          # known expectation of h(X) under Uniform(a,b)
beta_cv = 1.0       # control variate coefficient (tune as needed)

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
    Target integrand f(x).

    Example: f(x) = x^2 on [0,1].
    """
    return x * x


def h_control(x):
    """
    Control variate function h(x) with known expectation mu_h.

    Example: h(x) = x, X ~ Uniform(0,1) => E[h] = 1/2.
    """
    return x


def main():
    rng = LCG(seed, A, C, M)

    with open(output_file, "w") as f_out:
        f_out.write("# Control variates vs crude MC\n")
        f_out.write("# col1: N\n")
        f_out.write("# col2: I_crude_N\n")
        f_out.write("# col3: Var_crude_N\n")
        f_out.write("# col4: I_cv_N\n")
        f_out.write("# col5: Var_cv_N\n")
        f_out.write("# col6: error_crude_N\n")
        f_out.write("# col7: error_cv_N\n")

        total_N = 0

        # Running stats for crude: Y = f(X)
        mean_f = 0.0
        M2_f = 0.0

        # Running stats for control variate: Z = f(X) - beta*(h(X)-mu_h)
        mean_Z = 0.0
        M2_Z = 0.0

        I_crude_prev = None
        I_cv_prev = None

        N_vals = []
        I_crude_vals = []
        Var_crude_vals = []
        I_cv_vals = []
        Var_cv_vals = []
        Err_crude_vals = []
        Err_cv_vals = []

        for N_target in N_list:
            while total_N < N_target:
                u = rng.rand()
                x = a + (b - a) * u

                y = f(x)
                h_val = h_control(x)
                Z = y - beta_cv * (h_val - mu_h)

                total_N += 1

                # update crude stats
                delta_f = y - mean_f
                mean_f += delta_f / float(total_N)
                delta2_f = y - mean_f
                M2_f += delta_f * delta2_f

                # update control variate stats
                delta_Z = Z - mean_Z
                mean_Z += delta_Z / float(total_N)
                delta2_Z = Z - mean_Z
                M2_Z += delta_Z * delta2_Z

            if total_N > 1:
                var_f = M2_f / float(total_N - 1)
                var_Z = M2_Z / float(total_N - 1)
            else:
                var_f = 0.0
                var_Z = 0.0

            # estimator of integral for both
            I_crude = (b - a) * mean_f
            Var_crude = (b - a) ** 2 * var_f / float(total_N)

            # Note: Z already has expectation I / (b-a) when scaled,
            # so integral estimate is:
            I_cv = (b - a) * mean_Z
            Var_cv = (b - a) ** 2 * var_Z / float(total_N)

            if I_crude_prev is None:
                err_crude = 0.0
            else:
                err_crude = abs(I_crude - I_crude_prev)

            if I_cv_prev is None:
                err_cv = 0.0
            else:
                err_cv = abs(I_cv - I_cv_prev)

            f_out.write(f"{total_N} {I_crude} {Var_crude} "
                        f"{I_cv} {Var_cv} {err_crude} {err_cv}\n")

            N_vals.append(total_N)
            I_crude_vals.append(I_crude)
            Var_crude_vals.append(Var_crude)
            I_cv_vals.append(I_cv)
            Var_cv_vals.append(Var_cv)
            Err_crude_vals.append(err_crude)
            Err_cv_vals.append(err_cv)

            I_crude_prev = I_crude
            I_cv_prev = I_cv

    print("Control variates data written to:", output_file)

    if HAVE_MPL:
        plt.figure()
        plt.plot(N_vals, Var_crude_vals, marker='o', label="Var(crude)")
        plt.plot(N_vals, Var_cv_vals, marker='s', label="Var(control)")
        plt.xlabel("N")
        plt.ylabel("Var[estimator]")
        plt.title("Control variates: variance estimates")
        plt.legend()

        eps = 1e-16
        err_c_no_zero = [e if e > 0.0 else eps for e in Err_crude_vals]
        err_cv_no_zero = [e if e > 0.0 else eps for e in Err_cv_vals]
        if len(N_vals) > 1:
            plt.figure()
            plt.semilogy(N_vals[1:], err_c_no_zero[1:], marker='o',
                         label="crude error")
            plt.semilogy(N_vals[1:], err_cv_no_zero[1:], marker='s',
                         label="control error")
            plt.xlabel("N")
            plt.ylabel("|I_N - I_prev|")
            plt.title("Control variates: numerical error")
            plt.legend()
        plt.show()


if __name__ == "__main__":
    main()