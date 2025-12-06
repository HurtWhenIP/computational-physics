"""
Gauss–Legendre quadrature (2, 3, 4, 5 points) on [a, b] with numerical error.

What this script does:
- Approximates I = ∫_a^b f(x) dx using n-point Gauss–Legendre quadrature.
- Supports n = 2, 3, 4, 5 (nodes and weights hard-coded for [-1, 1]).
- Maps general interval [a, b] to [-1, 1] using:
      x = (b - a)/2 * t + (a + b)/2
      dx = (b - a)/2 dt
- For each chosen n:
    * I_n      : n-point Gauss–Legendre approximation.
    * error_n  : |I_n - I_prev|, purely numerical difference with previous n.
- Writes data to 'gauss_legendre_convergence.dat'.
- Optionally plots results with matplotlib if available.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * a, b             : limits of integration.
    * n_list           : list of orders (subset of [2, 3, 4, 5]).
- Step 2: Change f(x) to your integrand.
- Step 3: Run:
        python3 gauss_legendre_quadrature.py
- Step 4: Use gnuplot:
        plot 'gauss_legendre_convergence.dat' using 1:2 with linespoints title 'I_n'
        plot 'gauss_legendre_convergence.dat' using 1:3 with linespoints title 'error'

Data file columns:
    col1: n        (number of Gauss–Legendre points)
    col2: I_n      (Gauss–Legendre approximation on [a, b])
    col3: error_n  (|I_n - I_prev|, purely numerical)
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = -1.0                    # Left limit of integration
b = 1.0                     # Right limit of integration
n_list = [2, 3, 4, 5]       # Orders of Gauss–Legendre to use
output_file = "gauss_legendre_convergence.dat"
# ===============================================================


def f(x):
    """
    Integrand f(x). Replace this with the given function in the exam.

    Example: f(x) = x^2
    """
    return x * x


def get_gauss_legendre_nodes_weights(n):
    """
    Return nodes and weights for n-point Gauss–Legendre quadrature on [-1, 1].

    The values are standard and hard-coded, taken from tables.
    We do NOT compute them dynamically.

    Returns:
        nodes  : list of t_k
        weights: list of w_k
    """
    if n == 2:
        # 2-point Gauss–Legendre on [-1, 1]
        nodes = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
        weights = [1.0, 1.0]
    elif n == 3:
        # 3-point Gauss–Legendre
        nodes = [0.0,
                 -math.sqrt(3.0 / 5.0),
                 math.sqrt(3.0 / 5.0)]
        weights = [8.0 / 9.0,
                   5.0 / 9.0,
                   5.0 / 9.0]
    elif n == 4:
        # 4-point Gauss–Legendre
        sqrt_30 = math.sqrt(30.0)
        nodes = [
            -math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
            -math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
             math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
             math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
        ]
        weights = [
            (18.0 - sqrt_30) / 36.0,
            (18.0 + sqrt_30) / 36.0,
            (18.0 + sqrt_30) / 36.0,
            (18.0 - sqrt_30) / 36.0
        ]
    elif n == 5:
        # 5-point Gauss–Legendre
        nodes = [
            0.0,
            -1.0/3.0 * math.sqrt(5.0 - 2.0 * math.sqrt(10.0/7.0)),
             1.0/3.0 * math.sqrt(5.0 - 2.0 * math.sqrt(10.0/7.0)),
            -1.0/3.0 * math.sqrt(5.0 + 2.0 * math.sqrt(10.0/7.0)),
             1.0/3.0 * math.sqrt(5.0 + 2.0 * math.sqrt(10.0/7.0))
        ]
        weights = [
            128.0 / 225.0,
            (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
            (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
            (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
            (322.0 - 13.0 * math.sqrt(70.0)) / 900.0
        ]
    else:
        raise ValueError("Only n = 2, 3, 4, 5 are supported.")

    return nodes, weights


def gauss_legendre_integral(a, b, n):
    """
    Compute ∫_a^b f(x) dx using n-point Gauss–Legendre quadrature.

    Steps:
    1. Get nodes t_k and weights w_k for [-1, 1].
    2. Map from t in [-1, 1] to x in [a, b] by:
           x = ((b - a) / 2) * t + (a + b) / 2
       and dx = (b - a) / 2 dt.
    3. Approximate integral as:
           I ≈ sum( w_k * f(x_k) ) * (b - a) / 2
    """

    nodes, weights = get_gauss_legendre_nodes_weights(n)

    # Half-width and midpoint of the interval
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)

    I = 0.0
    for t, w in zip(nodes, weights):
        x = half * t + mid
        I += w * f(x)

    # Multiply by the scaling factor from dx
    I *= half
    return I


def main():
    fout = open(output_file, "w")
    fout.write("# col1: n (number of Gauss–Legendre points)\n")
    fout.write("# col2: I_n (Gauss–Legendre approximation on [a, b])\n")
    fout.write("# col3: error_n = |I_n - I_prev| (purely numerical)\n")

    I_prev = None
    n_vals = []
    I_vals = []
    err_vals = []

    for n in n_list:
        I_n = gauss_legendre_integral(a, b, n)

        if I_prev is None:
            error = 0.0
        else:
            error = abs(I_n - I_prev)

        fout.write(f"{n} {I_n} {error}\n")

        n_vals.append(n)
        I_vals.append(I_n)
        err_vals.append(error)

        I_prev = I_n

    fout.close()

    print("Gauss–Legendre quadrature study finished.")
    print("Results written to:", output_file)
    print("Last approximation I_n =", I_prev)
    print("Example gnuplot commands:")
    print("  plot 'gauss_legendre_convergence.dat' using 1:2 with linespoints title 'I_n'")
    print("  plot 'gauss_legendre_convergence.dat' using 1:3 with linespoints title 'error'")

    if HAVE_MPL:
        # Plot I_n vs n
        plt.figure()
        plt.plot(n_vals, I_vals, marker='o')
        plt.xlabel("n (Gauss–Legendre points)")
        plt.ylabel("I_n")
        plt.title("Gauss–Legendre: integral vs n")

        # Log-plot for numerical error
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]

        plt.figure()
        plt.semilogy(n_vals, err_no_zero, marker='o')
        plt.xlabel("n (Gauss–Legendre points)")
        plt.ylabel("|I_n - I_prev|")
        plt.title("Gauss–Legendre: numerical error (log scale)")

        plt.show()


if __name__ == "__main__":
    main()