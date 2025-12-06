"""
Linear Congruential Generator (LCG) – pseudo-random numbers in (0,1).

What this script does:
- Implements a simple LCG:
      X_{n+1} = (a * X_n + c) mod m
  and returns U_n = X_n / m in (0,1).
- Generates N uniform(0,1) numbers.
- Writes them to 'lcg_sequence.dat' for inspection.
- Optionally plots the sequence and histogram with matplotlib (if available).

How to use in the exam:
- Step 1: Adjust USER PARAMETERS:
      seed, a, c, m, N.
- Step 2: Run:
      python3 mc_lcg.py
- Step 3: Gnuplot examples:
      plot 'lcg_sequence.dat' using 1:2 with linespoints title 'U_n'
      plot 'lcg_sequence.dat' using 2:(1.0) smooth kdensity title 'density'

Data file columns:
    col1: n   (index, starting from 0)
    col2: U_n (uniform(0,1) pseudo-random number)
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
seed = 12345          # initial seed X_0
a    = 1103515245     # multiplier
c    = 12345          # increment
m    = 2**31 - 1      # modulus (large prime-ish)
N    = 1000           # how many random numbers to generate
output_file = "lcg_sequence.dat"
# ===============================================================


class LCG:
    """
    Simple Linear Congruential Generator.

    Usage:
        rng = LCG(seed, a, c, m)
        u = rng.rand()  # uniform in (0,1)
    """

    def __init__(self, seed, a, c, m):
        self.X = seed
        self.a = a
        self.c = c
        self.m = m

    def rand(self):
        """Return next U in (0,1)."""
        self.X = (self.a * self.X + self.c) % self.m
        return self.X / float(self.m)


def main():
    rng = LCG(seed, a, c, m)

    with open(output_file, "w") as f:
        f.write("# LCG uniform(0,1) sequence\n")
        f.write("# col1: n (index), col2: U_n\n")
        for n in range(N):
            u = rng.rand()
            f.write(f"{n} {u}\n")

    print("LCG sequence written to:", output_file)
    print("Example gnuplot:")
    print("  plot 'lcg_sequence.dat' using 1:2 with linespoints title 'U_n'")

    if HAVE_MPL:
        # Read back for plotting
        ns = []
        us = []
        with open(output_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                ns.append(int(parts[0]))
                us.append(float(parts[1]))

        # Plot sequence
        plt.figure()
        plt.plot(ns, us, marker='.', linestyle='none')
        plt.xlabel("n")
        plt.ylabel("U_n")
        plt.title("LCG uniform sequence")

        # Histogram
        plt.figure()
        plt.hist(us, bins=20, edgecolor='black')
        plt.xlabel("U")
        plt.ylabel("Count")
        plt.title("LCG histogram")
        plt.show()


if __name__ == "__main__":
    main()