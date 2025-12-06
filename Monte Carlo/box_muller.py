"""
Box–Muller transform – generate standard normal N(0,1) from uniform(0,1).

Algorithm (basic Box–Muller):
- Generate U1, U2 ~ Uniform(0,1).
- Compute:
      R  = sqrt(-2 ln U1)
      T  = 2 pi U2
      Z1 = R cos T
      Z2 = R sin T
- Z1 and Z2 are i.i.d. standard normal N(0,1).

What this script does:
- Uses an LCG to create U1, U2.
- Generates N normal samples (using both Z1 and Z2 per pair).
- Tracks running mean and variance, with numerical convergence:
      error_N = |mean_N - mean_{prev}|.
- Writes 'box_muller.dat'.
- Also stores all generated Zs and, if matplotlib is available,
  plots a histogram of the obtained normal distribution.

How to use in the exam:
- Edit USER PARAMETERS: N_pairs (each pair gives 2 normals).
- Run:
      python3 mc_box_muller.py
- Gnuplot:
      plot 'box_muller.dat' using 1:3 with linespoints title 'mean_N'
      plot 'box_muller.dat' using 1:5 with linespoints title 'error'

Data file columns:
    col1: N_normals
    col2: last_Z
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

N_pairs = 1000          # total normals = 2*N_pairs
output_file = "box_muller.dat"
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
        f.write("# Box–Muller standard normal generator\n")
        f.write("# col1: N_normals\n")
        f.write("# col2: last_Z\n")
        f.write("# col3: mean_N\n")
        f.write("# col4: var_N\n")
        f.write("# col5: error_N = |mean_N - mean_prev|\n")

        N_vals = []
        mean_vals = []
        var_vals = []
        err_vals = []

        total_N = 0
        mean = 0.0
        M2 = 0.0
        prev_mean = None

        last_z = 0.0

        # store all normals for distribution plot
        normals = []

        for _ in range(N_pairs):
            # two uniforms
            u1 = rng.rand()
            u2 = rng.rand()
            if u1 == 0.0:
                u1 = 1e-16

            R = math.sqrt(-2.0 * math.log(u1))
            T = 2.0 * math.pi * u2

            z1 = R * math.cos(T)
            z2 = R * math.sin(T)

            for z in (z1, z2):
                total_N += 1
                last_z = z
                normals.append(z)

                delta = z - mean
                mean += delta / float(total_N)
                delta2 = z - mean
                M2 += delta * delta2

            if total_N > 1:
                var = M2 / float(total_N - 1)
            else:
                var = 0.0

            if prev_mean is None:
                error = 0.0
            else:
                error = abs(mean - prev_mean)

            f.write(f"{total_N} {last_z} {mean} {var} {error}\n")

            N_vals.append(total_N)
            mean_vals.append(mean)
            var_vals.append(var)
            err_vals.append(error)

            prev_mean = mean

    print("Box–Muller data written to:", output_file)

    if HAVE_MPL:
        # Histogram of generated normals
        plt.figure()
        n, bins, _ = plt.hist(normals, bins=30, edgecolor="black")
        plt.xlabel("z")
        plt.ylabel("Count")
        plt.title("Box–Muller: N(0,1) samples")

        # ===== Overlay standard normal pdf with correct scaling =====
        bin_width = bins[1] - bins[0]
        xmin = bins[0]
        xmax = bins[-1]
        xr = [xmin + (xmax - xmin) * i / 500.0 for i in range(501)]
        pdf = [(1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x) for x in xr]
        scale = len(normals) * bin_width
        pdf_scaled = [p * scale for p in pdf]
        plt.plot(xr, pdf_scaled, "r-", linewidth=2, label="target pdf")
        plt.legend()

        # --- Convergence plot: skip first zero-error point ---
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]
        plt.figure()
        plt.semilogy(N_vals[1:], err_no_zero[1:], marker="o")
        plt.xlabel("N_normals")
        plt.ylabel("|mean_N - mean_prev|")
        plt.title("Box–Muller: numerical convergence of mean")

        plt.show()


if __name__ == "__main__":
    main()