"""
D-dimensional Monte Carlo integration over a hypercube.

Goal:
    Approximate
        I = ∫_Domain f(x_1, x_2, ..., x_D) dV
    where Domain = [a, b]^D (hypercube).

Method:
    - Sample X_i ~ Uniform([a, b]^D) independently.
    - Evaluate Y_i = f(X_i).
    - Let:
          mean_f(N) = (1/N) Σ Y_i
          Volume    = (b - a)^D
          I_N       = Volume * mean_f(N)
    - Estimate variance numerically:
          s_f^2(N)  = sample variance of Y_i
          Var[I_N] ≈ Volume^2 * s_f^2(N) / N

Numerical error:
    - error_N = |I_N - I_prev| between successive N in N_list.

Outputs:
    - 'mc_ddim.dat'

        col1: N
        col2: I_N
        col3: Var_I_N
        col4: error_N

How to use in the exam:
    - Set USER PARAMETERS:
          D, a, b, N_list, and define f(vec_x) for a D-dimensional integrand.
    - Run:
          python3 mc_integration_d_dim.py
    - Gnuplot:
          plot 'mc_ddim.dat' using 1:2 with linespoints title 'I_N'
          plot 'mc_ddim.dat' using 1:4 with linespoints title 'error'
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
D = 5                 # dimension of the integral
a = 0.0               # lower bound for each dimension
b = 1.0               # upper bound for each dimension
N_list = [100, 500, 1000, 5000]
output_file = "mc_ddim.dat"

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


def f(vec_x):
    """
    D-dimensional integrand f(x1, x2, ..., xD).

    vec_x is a Python list [x1, x2, ..., xD].

    Example: f(x) = exp(-(x1^2 + ... + xD^2))
    """
    s2 = 1
    for xi in vec_x:
        s2 *= xi
    return s2  # EDIT in exam if needed


def main():
    rng = LCG(seed, A, C, M)
    Volume = (b - a) ** D

    with open(output_file, "w") as f_out:
        f_out.write("# D-dimensional Monte Carlo integration over [a,b]^D\n")
        f_out.write("# col1: N (samples)\n")
        f_out.write("# col2: I_N (MC estimate)\n")
        f_out.write("# col3: Var_I_N (MC variance estimate)\n")
        f_out.write("# col4: error_N = |I_N - I_prev|\n")

        I_prev = None
        N_vals = []
        I_vals = []
        Var_vals = []
        Err_vals = []

        # Running stats for Y = f(X)
        total_N = 0
        mean_f = 0.0
        M2_f = 0.0

        for N_target in N_list:
            while total_N < N_target:
                # sample a D-dimensional point
                x_vec = []
                for _ in range(D):
                    u = rng.rand()
                    x = a + (b - a) * u
                    x_vec.append(x)

                y = f(x_vec)

                total_N += 1
                # Welford update
                delta = y - mean_f
                mean_f += delta / float(total_N)
                delta2 = y - mean_f
                M2_f += delta * delta2

            if total_N > 1:
                var_f = M2_f / float(total_N - 1)
            else:
                var_f = 0.0

            I_N = Volume * mean_f
            Var_I_N = Volume ** 2 * var_f / float(total_N)

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

    print("D-dimensional MC data written to:", output_file)

    if HAVE_MPL:
        plt.figure()
        plt.plot(N_vals, I_vals, marker='o')
        plt.xlabel("N")
        plt.ylabel("I_N")
        plt.title(f"MC {D}D: estimate vs N")

        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in Err_vals]
        if len(N_vals) > 1:
            plt.figure()
            plt.semilogy(N_vals[1:], err_no_zero[1:], marker='o')
            plt.xlabel("N")
            plt.ylabel("|I_N - I_prev|")
            plt.title(f"MC {D}D: numerical error")
        plt.show()


if __name__ == "__main__":
    main()