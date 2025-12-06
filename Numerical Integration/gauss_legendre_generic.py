"""
Gauss–Legendre quadrature with generic n-point rule and numerical error analysis.

What this script does:
- Approximates I = ∫_a^b f(x) dx using n-point Gauss–Legendre quadrature.
- Works for ANY integer n >= 2 by:
    * Computing Legendre polynomial P_n(x) and its derivative via recurrence.
    * Finding roots of P_n(x) on (-1,1) using Newton's method with good initial
      guesses that exploit symmetry.
    * Computing corresponding weights:
            w_i = 2 / [ (1 - x_i^2) * (P_n'(x_i))^2 ].
- Maps nodes from [-1,1] to [a,b]:
      x = (b-a)/2 * t + (a+b)/2
      I_n = (b-a)/2 * Σ w_i f(x_i)

Convergence / error (purely numerical):
- For n values in N_list (sorted), compute:
      I_n
- For each n (except the first), define numerical error:
      error_n = |I_n - I_prev|,
  where I_prev is the integral for the previous n in N_list.

Output:
- Writes 'gauss_legendre_convergence.dat' with:
    col1: n_points
    col2: I_n
    col3: error_n = |I_n - I_prev|

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
      * a, b        : integration limits
      * N_list      : list of n-point rules to use, e.g., [2, 3, 4, 8, 16]
      * output_file : filename for .dat output
- Step 2: Edit f(x) to match the given integrand.
- Step 3: Run:
      python3 gauss_legendre_generic.py
- Step 4 (gnuplot examples):
      plot 'gauss_legendre_convergence.dat' using 1:2 with linespoints title 'I_n'
      plot 'gauss_legendre_convergence.dat' using 1:3 with linespoints title 'error'

Matplotlib (if available):
- Plot I_n vs n.
- Plot error_n vs n (on a log scale), skipping the first zero-error point.

Note:
- All error measures are purely numerical (difference between successive n),
  NO analytic true integral is used anywhere.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = -1.0                        # left limit of integration
b =  1.0                        # right limit of integration
N_list = [2, 3, 4, 5, 16]       # list of n-points to use (any integers >= 2)
output_file = "gauss_legendre_convergence.dat"
# ===============================================================


def f(x):
    """
    Integrand f(x). Replace this with the given function in the exam.

    Example here:
        f(x) = exp(-x^2)
    """
    return x*x


# ----------------------------------------------------------------------
# Legendre polynomial P_n(x) and derivative using recurrence relations
# ----------------------------------------------------------------------

def legendre_Pn_and_derivative(n, x):
    """
    Compute P_n(x) and P_n'(x) using recurrence.

    Recurrence for polynomials:
        P_0(x) = 1
        P_1(x) = x
        (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x), for k = 1..n-1.

    Derivative:
        P_n'(x) = n / (x^2 - 1) * [x P_n(x) - P_{n-1}(x)].

    Returns:
        (Pn, dPn)
    """
    # Base cases
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return x, 1.0

    P_nm1 = 1.0    # P_0
    P_n_  = x      # P_1

    for k in range(1, n):
        # Compute P_{k+1} from P_k and P_{k-1}
        P_np1 = ((2.0 * k + 1.0) * x * P_n_ - k * P_nm1) / (k + 1.0)
        P_nm1, P_n_ = P_n_, P_np1

    # Now P_n_ = P_n, P_nm1 = P_{n-1}
    Pn = P_n_
    Pnm1 = P_nm1

    # Derivative using the standard relation
    if abs(x * x - 1.0) < 1e-14:
        # avoid division by zero; near endpoints derivative is not used in Newton
        dPn = 0.0
    else:
        dPn = (n * (x * Pn - Pnm1)) / (x * x - 1.0)

    return Pn, dPn


# ----------------------------------------------------------------------
# Find Gauss–Legendre nodes and weights for generic n
# ----------------------------------------------------------------------

def gauss_legendre_nodes_weights(n, tol=1e-14, max_iter=100):
    """
    Compute nodes and weights for n-point Gauss–Legendre quadrature.

    Approach:
    - Exploit symmetry: roots of P_n(x) lie in (-1,1) and come in ± pairs.
    - Only compute the positive roots, then reflect to negative.
    - Initial guesses:
          x_k^(0) ≈ cos(π (4k - 1)/(4n + 2)), for k = 1..m
      where m = n//2.
    - Use Newton's method to refine each root:
          x_{new} = x - P_n(x)/P_n'(x).

    Weights:
        w_i = 2 / [ (1 - x_i^2) * (P_n'(x_i))^2 ]
    """
    if n < 2:
        raise ValueError("n must be >= 2 for Gauss–Legendre quadrature.")

    m = n // 2          # number of positive roots
    nodes = [0.0] * n
    weights = [0.0] * n

    for k in range(1, m + 1):
        # initial guess for k-th positive root
        x = math.cos(math.pi * (4.0 * k - 1.0) / (4.0 * n + 2.0))

        # Newton iterations
        for _ in range(max_iter):
            Pn, dPn = legendre_Pn_and_derivative(n, x)
            if abs(dPn) < 1e-16:
                break
            dx = -Pn / dPn
            x_new = x + dx
            if abs(dx) < tol:
                x = x_new
                break
            x = x_new

        # compute final Pn'(x) for weight
        Pn, dPn = legendre_Pn_and_derivative(n, x)
        w = 2.0 / ((1.0 - x * x) * dPn * dPn)

        # store positive root and its negative counterpart
        nodes[k - 1] = -x
        nodes[n - k] = x
        weights[k - 1] = w
        weights[n - k] = w

    # If n is odd, there is a root at x = 0
    if n % 2 == 1:
        # central index
        idx = n // 2
        x = 0.0
        Pn, dPn = legendre_Pn_and_derivative(n, x)
        # note: for odd n, P_n is odd, so P_n(0)=0, the derivative is nonzero.
        w = 2.0 / ((1.0 - x * x) * dPn * dPn)
        nodes[idx] = x
        weights[idx] = w

    return nodes, weights


def gauss_legendre_integral(a, b, n):
    """
    Compute I_n = ∫_a^b f(x) dx using n-point Gauss–Legendre quadrature.

    Steps:
    - Get nodes t_i and weights w_i on [-1,1].
    - Map t_i to x_i in [a,b]:
          x_i = 0.5 * (b - a) * t_i + 0.5 * (a + b)
    - Integral:
          I_n = 0.5 * (b - a) * Σ w_i f(x_i)
    """
    nodes, weights = gauss_legendre_nodes_weights(n)

    factor = 0.5 * (b - a)
    shift = 0.5 * (a + b)

    total = 0.0
    for t, w in zip(nodes, weights):
        x = factor * t + shift
        total += w * f(x)

    I_n = factor * total
    return I_n


def main():
    # Sort N_list so error is between successive n in ascending order
    sorted_N = sorted(N_list)

    with open(output_file, "w") as fout:
        fout.write("# Gauss–Legendre quadrature (generic n-point rule)\n")
        fout.write("# col1: n_points\n")
        fout.write("# col2: I_n (integral approximation)\n")
        fout.write("# col3: error_n = |I_n - I_prev|\n")

        I_prev = None
        n_vals = []
        I_vals = []
        err_vals = []

        for n in sorted_N:
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

    print("Gauss–Legendre convergence data written to:", output_file)
    print("Example gnuplot commands:")
    print("  plot 'gauss_legendre_convergence.dat' using 1:2 with linespoints title 'I_n'")
    print("  plot 'gauss_legendre_convergence.dat' using 1:3 with linespoints title 'error'")

    # ---------------- Matplotlib sanity plots ----------------
    if HAVE_MPL:
        # I_n vs n
        plt.figure()
        plt.plot(n_vals, I_vals, marker='o')
        plt.xlabel("n (points)")
        plt.ylabel("I_n")
        plt.title("Gauss–Legendre: integral vs n")

        # error vs n (skip first zero-error point)
        eps = 1e-16
        err_no_zero = [e if e > 0.0 else eps for e in err_vals]
        if len(n_vals) > 1:
            plt.figure()
            plt.semilogy(n_vals[1:], err_no_zero[1:], marker='o')
            plt.xlabel("n (points)")
            plt.ylabel("|I_n - I_prev|")
            plt.title("Gauss–Legendre: numerical error")

        plt.show()


if __name__ == "__main__":
    main()