"""
General improper integration using substitutions to a finite interval
and composite Simpson's 1/3 rule.

What this script does:
----------------------
Approximates integrals of the form:

1) Finite limits:
       I = ∫_a^b f(x) dx

2) One-sided improper (a to +∞):
       I = ∫_a^∞ f(x) dx

3) One-sided improper (-∞ to b):
       I = ∫_-∞^b f(x) dx

4) Two-sided improper (-∞ to +∞):
       I = ∫_-∞^∞ f(x) dx

by mapping them to an integral over a finite t-interval and then applying
**composite Simpson's 1/3 rule** in t.

Substitutions used (you can quote these in the exam):
----------------------------------------------------
Let g(t) be the transformed integrand after substitution.

Case 1: FINITE
    x = t                    with t ∈ [a, b]
    => dx/dt = 1
    => g(t) = f(t)

Case 2: A_INF  (a to +∞)
    x = a + t / (1 - t),     t ∈ [0, 1)
    dx/dt = 1 / (1 - t)^2
    => ∫_a^∞ f(x) dx = ∫_0^1 f(a + t/(1-t)) * [1 / (1 - t)^2] dt
    Numerically, we integrate t ∈ [0, 1 - delta] to avoid t = 1.

Case 3: MINF_B (-∞ to b)
    x = b - t / (1 - t),     t ∈ [0, 1)
    dx/dt = 1 / (1 - t)^2
    => ∫_-∞^b f(x) dx = ∫_0^1 f(b - t/(1-t)) * [1 / (1 - t)^2] dt
    Numerically, we integrate t ∈ [0, 1 - delta].

Case 4: MINF_INF (-∞ to +∞)
    x = tan(π (t - 1/2)),   t ∈ (0, 1)
    dx/dt = π sec^2(π (t - 1/2))
    => ∫_-∞^∞ f(x) dx = ∫_0^1 f(tan(π (t-1/2))) * π sec^2(π (t-1/2)) dt
    Numerically, we integrate t ∈ [delta, 1 - delta] to avoid t=0,1.

IMPORTANT:
- delta is a small cutoff (e.g., 1e-6) to stay away from singularities
  at t=0 or t=1. The tails are truncated numerically. You can reduce
  delta to improve accuracy if needed.

Convergence / error (purely numerical):
---------------------------------------
We do a step-refinement study with different N (even) values:

- For each N in N_list (e.g., [20, 40, 80, 160]):
      I_N = Simpson's 1/3 approximation in t.
- Numerical error:
      error_N = |I_N - I_prev|
  where I_prev is the integral from the previous N value in N_list.

No "true" or analytic value of the integral is ever used.

Output file:
------------
- 'improper_general_convergence.dat'

    col1: N        (number of t-subintervals, must be even)
    col2: h        (t step size)
    col3: I_N      (approximation of the original integral)
    col4: error_N  (|I_N - I_prev|, purely numerical)

How to use in the exam:
-----------------------
1) Choose the type of integral by setting:

       improper_type = "FINITE"   (∫_a^b)
                      = "A_INF"   (∫_a^∞)
                      = "MINF_B"  (∫_-∞^b)
                      = "MINF_INF" (∫_-∞^∞)

2) Set a and/or b:
       - For FINITE  : a and b are both finite.
       - For A_INF   : a is finite, b is ignored.
       - For MINF_B  : b is finite, a is ignored.
       - For MINF_INF: both a and b are ignored.

3) Edit f(x) to the given integrand.

4) Edit N_list for the number of t-subintervals (MUST be even).

5) Run:

       python3 improper_integration_general.py

6) In gnuplot:

       plot 'improper_general_convergence.dat' using 1:3 with linespoints \
            title 'I_N'

       plot 'improper_general_convergence.dat' using 1:4 with linespoints \
            title 'error'

Matplotlib (optional sanity checks):
------------------------------------
If matplotlib is available, the script will also:

- Plot I_N vs N (integral vs resolution).
- Plot error_N vs N (log scale), skipping the first zero-error point.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
# TYPE OF IMPROPER INTEGRAL:
# "FINITE", "A_INF", "MINF_B", "MINF_INF"
improper_type = "MINF_INF"

# Finite endpoints (used depending on improper_type):
a = 0.0     # used for FINITE and A_INF
b = 1.0     # used for FINITE and MINF_B

# List of even N values (number of t-subintervals).
# Larger N => finer t-grid.
N_list = [20, 40, 80, 160]

# Small cutoff to avoid singular t-endpoints for improper cases
delta = 1.0e-6

output_file = "improper_general_convergence.dat"
# ===============================================================


def f(x):
    """
    Integrand f(x). Replace with the given function.

    EXAMPLES (just one at a time in the exam):
    ------------------------------------------
    - For ∫_0^∞ e^{-x} dx, use:
          return math.exp(-x)

    - For ∫_-∞^∞ e^{-x^2} dx, use:
          return math.exp(-x*x)
    """
    return 1 / (1 + x*x)  # EDIT THIS IN THE EXAM


# ----------------------------------------------------------------------
# t-domain and transformed integrand g(t) depending on improper_type
# ----------------------------------------------------------------------

def t_interval_and_g():
    """
    Return:
        t_left, t_right, g(t) as a function handle.

    t_left, t_right:
        The finite t-interval over which we integrate using Simpson.

    g(t):
        The transformed integrand such that:

            Original integral = ∫_{t_left}^{t_right} g(t) dt

    depending on improper_type:
        - "FINITE"
        - "A_INF"
        - "MINF_B"
        - "MINF_INF"
    """

    # Case 1: FINITE integral ∫_a^b f(x) dx
    if improper_type == "FINITE":
        def g(t):
            # Here we simply set x = t, dx/dt = 1
            return f(t)
        return a, b, g

    # Case 2: a to +∞ : ∫_a^∞ f(x) dx
    elif improper_type == "A_INF":
        t_left = 0.0
        t_right = 1.0 - delta  # avoid t=1

        def g(t):
            # x = a + t/(1-t), dx/dt = 1/(1-t)^2
            if t >= 1.0:
                return 0.0
            x = a + t / (1.0 - t)
            dxdt = 1.0 / ((1.0 - t) * (1.0 - t))
            return f(x) * dxdt

        return t_left, t_right, g

    # Case 3: -∞ to b : ∫_-∞^b f(x) dx
    elif improper_type == "MINF_B":
        t_left = 0.0
        t_right = 1.0 - delta  # avoid t=1

        def g(t):
            # x = b - t/(1-t), dx/dt = 1/(1-t)^2
            if t >= 1.0:
                return 0.0
            x = b - t / (1.0 - t)
            dxdt = 1.0 / ((1.0 - t) * (1.0 - t))
            return f(x) * dxdt

        return t_left, t_right, g

    # Case 4: -∞ to +∞ : ∫_-∞^∞ f(x) dx
    elif improper_type == "MINF_INF":
        t_left = delta
        t_right = 1.0 - delta  # avoid t=0 and t=1

        def g(t):
            # x = tan(pi (t - 1/2)), dx/dt = pi sec^2(pi (t-1/2))
            u = math.pi * (t - 0.5)
            x = math.tan(u)
            dxdt = math.pi / (math.cos(u) * math.cos(u))  # pi sec^2(u)
            return f(x) * dxdt

        return t_left, t_right, g

    else:
        raise ValueError("Unknown improper_type: " + improper_type)


# ----------------------------------------------------------------------
# Composite Simpson's 1/3 rule on [t_left, t_right]
# ----------------------------------------------------------------------

def simpson_one_third_general(t_left, t_right, N, g):
    """
    Composite Simpson's 1/3 rule on [t_left, t_right] with N subintervals.

    REQUIREMENT:
        N must be even.

    Formula:
        h = (t_right - t_left) / N
        I ≈ (h/3) [g(t0) + g(tN)
                   + 4 * sum_{odd i} g(t_i)
                   + 2 * sum_{even i (except endpoints)} g(t_i) ]
        where t_i = t_left + i*h, i=0..N.
    """
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's 1/3 rule.")

    h = (t_right - t_left) / float(N)

    # endpoints
    s = g(t_left) + g(t_right)

    # odd indices
    for i in range(1, N, 2):
        t = t_left + i * h
        s += 4.0 * g(t)

    # even indices (excluding endpoints)
    for i in range(2, N, 2):
        t = t_left + i * h
        s += 2.0 * g(t)

    I = (h / 3.0) * s
    return I, h


# ----------------------------------------------------------------------
# Main driver: convergence in N using purely numerical error
# ----------------------------------------------------------------------

def main():
    # Get t-interval and transformed integrand
    t_left, t_right, g = t_interval_and_g()

    # Sort N_list so that error is between successive resolutions
    N_sorted = sorted(N_list)

    with open(output_file, "w") as fout:
        fout.write("# General improper integration via substitution and Simpson 1/3\n")
        fout.write("# improper_type = %s\n" % improper_type)
        fout.write("# col1: N (even, # of t-subintervals)\n")
        fout.write("# col2: h (t step size)\n")
        fout.write("# col3: I_N (approx integral)\n")
        fout.write("# col4: error_N = |I_N - I_prev|\n")

        I_prev = None
        N_vals = []
        I_vals = []
        Err_vals = []
        h_vals = []

        for N in N_sorted:
            I_N, h = simpson_one_third_general(t_left, t_right, N, g)

            if I_prev is None:
                error = 0.0
            else:
                error = abs(I_N - I_prev)

            fout.write(f"{N} {h} {I_N} {error}\n")

            N_vals.append(N)
            h_vals.append(h)
            I_vals.append(I_N)
            Err_vals.append(error)

            I_prev = I_N

    print("Approximate value of the improper integral =", I_vals[-1])

    print("Improper integral convergence data written to:", output_file)
    print("Example gnuplot commands:")
    print("  plot 'improper_general_convergence.dat' using 1:3 with linespoints title 'I_N'")
    print("  plot 'improper_general_convergence.dat' using 1:4 with linespoints title 'error'")

    # -------------------- Matplotlib sanity plots --------------------
    if HAVE_MPL and len(N_vals) > 0:
        # I_N vs N
        plt.figure()
        plt.plot(N_vals, I_vals, marker='o')
        plt.xlabel("N (even)")
        plt.ylabel("I_N")
        plt.title("Improper integral: Simpson 1/3 vs N")

        # error vs N (skip first zero-error point)
        if len(N_vals) > 1:
            eps = 1e-16
            err_no_zero = [e if e > 0.0 else eps for e in Err_vals]
            plt.figure()
            plt.semilogy(N_vals[1:], err_no_zero[1:], marker='o')
            plt.xlabel("N (even)")
            plt.ylabel("|I_N - I_prev|")
            plt.title("Improper integral: numerical error in N")

        plt.show()


if __name__ == "__main__":
    main()