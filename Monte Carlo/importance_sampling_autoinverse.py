"""
General importance sampling template with automatic inverse CDF.

Goal:
    Estimate the integral
        I = ∫_{q_a}^{q_b} f(x) dx
    using importance sampling with a user-defined proposal pdf q(x).

You ONLY edit:
    - f(x)      : target integrand
    - q_pdf(x)  : proposal pdf (unnormalized is fine)
    - q_a, q_b  : support of q(x)

Everything else is automatic:
    - Numerical normalization of q(x).
    - Numerical construction of CDF via trapezoidal rule.
    - Numerical inverse CDF sampling (piecewise linear).
    - Importance-sampling weights w = f(x)/q(x).
    - Integral estimate, variance, standard error.
    - .dat output for gnuplot.
    - Optional matplotlib sanity plots.

Constraints:
    - Uses only standard library + math + random (+ optional matplotlib).
    - No NumPy, SciPy, etc.

How to use in the exam:
-----------------------
1) Set USER PARAMETERS:
       q_a, q_b      : integration / proposal support.
       N_cdf_points  : grid points for building CDF (e.g., 1000).
       N_samples     : number of importance samples (e.g., 50000).

2) Edit f(x) to your integrand.

3) Edit q_pdf(x) so that q(x) > 0 on (q_a, q_b) and is integrable.
   You can give q_pdf unnormalized; the code normalizes it.

4) Run:
       python3 importance_sampling_general.py

5) gnuplot examples:
       plot 'importance_samples.dat' using 1:2 with points title 'weights'

Data file columns:
    col1: x_i   (sample from q)
    col2: w_i   (importance weight = f(x_i)/q(x_i))
"""

import math
import random

# ============================================================
# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM =========
# Integration / proposal support
q_a = 0.0          # left end of support
q_b = 1.0          # right end of support

# Grid for building CDF (more points -> smoother inverse CDF)
N_cdf_points = 1000

# Number of importance samples
N_samples = 50000

output_file = "importance_samples.dat"
# ============================================================


# ============================================================
# Target integrand f(x)   (YOU CHANGE THIS)
# ============================================================

def f(x):
    """
    Target integrand whose integral over [q_a, q_b] we want:

        I = ∫_{q_a}^{q_b} f(x) dx

    You ONLY change this function in the exam.

    Example 1 (default here):
        f(x) = exp(-x),   integral on [0,1] = 1 - e^{-1}.

    Example 2:
        f(x) = x*x

    Remember: the code never uses any analytic exact value, only
    the function evaluations.
    """
    return math.exp(-x)


# ============================================================
# Proposal pdf q(x)   (YOU CAN CHANGE THIS TOO)
# ============================================================

def q_pdf_raw(x):
    """
    Unnormalized proposal pdf q_raw(x).

    The code will:
      - numerically integrate q_raw(x) over [q_a, q_b] to get Z,
      - define normalized q(x) = q_raw(x)/Z.

    Requirements:
      - q_raw(x) >= 0 on [q_a, q_b],
      - not identically zero.

    Example default: triangular pdf on [0,1] proportional to 2x:

        q(x) ∝ 2x on [0,1].

    You can replace this by any other shape you like (e.g. exp(-x),
    x^2, etc.) as long as it is non-negative on [q_a, q_b].
    """
    if x < q_a or x > q_b:
        return 0.0
    # Default: triangular shape on [0,1] (assuming q_a=0, q_b=1)
    return 2.0 * x


# ============================================================
# Build normalized q(x) and its CDF on a grid
# ============================================================

def build_normalized_q_and_cdf():
    """
    Build:
        - xs   : grid of x in [q_a, q_b]
        - qnorm: normalized pdf values on the grid
        - cdf  : cumulative distribution on the grid, from 0 to 1.

    Uses trapezoidal rule numerically to:
        1) compute normalization constant Z for q_raw,
        2) build cumulative integrals of q(x) = q_raw(x)/Z.

    This CDF grid will then be used to sample from q via inverse CDF.
    """
    xs = []
    q_raw_vals = []

    # Step size on x-grid for CDF construction
    h = (q_b - q_a) / float(N_cdf_points - 1)

    # 1) build raw pdf samples
    for i in range(N_cdf_points):
        x = q_a + i * h
        xs.append(x)
        q_raw_vals.append(max(0.0, q_pdf_raw(x)))   # enforce non-negativity

    # 2) compute normalization constant Z via trapezoidal rule
    Z = 0.0
    for i in range(N_cdf_points - 1):
        Z += 0.5 * (q_raw_vals[i] + q_raw_vals[i+1]) * h

    if Z <= 0.0:
        raise ValueError("Proposal pdf q_pdf_raw integrates to zero or negative. "
                         "Check q_pdf_raw and [q_a, q_b].")

    # 3) normalized pdf on grid
    qnorm = [val / Z for val in q_raw_vals]

    # 4) build cumulative distribution using trapezoidal rule
    cdf = [0.0] * N_cdf_points
    cumulative = 0.0
    cdf[0] = 0.0
    for i in range(1, N_cdf_points):
        cumulative += 0.5 * (qnorm[i-1] + qnorm[i]) * h
        cdf[i] = cumulative

    # Due to numerical rounding, cdf[-1] may be slightly != 1; rescale
    final_cum = cdf[-1]
    if final_cum <= 0.0:
        raise ValueError("Final cumulative probability is non-positive; "
                         "check q_pdf_raw.")
    for i in range(N_cdf_points):
        cdf[i] /= final_cum

    return xs, qnorm, cdf, Z


# ============================================================
# Inverse CDF sampler (piecewise linear)
# ============================================================

def sample_from_q(xs, cdf):
    """
    Sample x from q(x) using the precomputed xs and cdf arrays.

    Algorithm:
        - Draw u ~ Uniform(0,1).
        - Find i such that cdf[i] <= u < cdf[i+1] (binary search).
        - Linearly interpolate x between xs[i] and xs[i+1].
    """
    u = random.random()

    # --- binary search on cdf ---
    lo = 0
    hi = len(cdf) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cdf[mid] <= u:
            lo = mid
        else:
            hi = mid

    # Now u in [cdf[lo], cdf[hi]], interpolate x
    if cdf[hi] == cdf[lo]:
        # Avoid division by zero; just pick midpoint
        return 0.5 * (xs[lo] + xs[hi])

    t = (u - cdf[lo]) / (cdf[hi] - cdf[lo])
    x = xs[lo] + t * (xs[hi] - xs[lo])
    return x


def q_pdf_normalized(x, Z):
    """
    Evaluate normalized proposal pdf q(x) = q_raw(x)/Z.
    (Used for weights.)

    Z is the normalization constant returned from build_normalized_q_and_cdf().
    """
    if x < q_a or x > q_b:
        return 0.0
    return q_pdf_raw(x) / Z


# ============================================================
# Importance sampling driver
# ============================================================

def importance_sampling():
    # Precompute q(x) and its CDF
    xs_grid, qnorm_grid, cdf_grid, Z = build_normalized_q_and_cdf()

    samples = []
    weights = []

    for _ in range(N_samples):
        x = sample_from_q(xs_grid, cdf_grid)
        qx = q_pdf_normalized(x, Z)
        if qx <= 0.0:
            # Should not happen if q_raw > 0 on support, but safe-guard anyway
            continue

        w = f(x) / qx
        samples.append(x)
        weights.append(w)

    n_eff = len(samples)
    if n_eff == 0:
        raise RuntimeError("No valid samples generated; check q_pdf_raw and support.")

    # Integral estimate
    mean_w = sum(weights) / n_eff

    # Sample variance of weights
    var_w = sum((w - mean_w)**2 for w in weights) / (n_eff - 1)
    std_error = math.sqrt(var_w / n_eff)

    # Write .dat file
    with open(output_file, "w") as fout:
        fout.write("# col1: x_i (sample from q)\n")
        fout.write("# col2: w_i = f(x_i)/q(x_i)   (importance weight)\n")
        for x, w in zip(samples, weights):
            fout.write(f"{x} {w}\n")

    print("\nImportance Sampling (general) results")
    print("-------------------------------------")
    print("Support [q_a, q_b] =", q_a, q_b)
    print("Samples used      =", n_eff)
    print("Integral estimate I ≈", mean_w)
    print("Std error               ≈", std_error)
    print("Data written to         :", output_file)
    print("Example gnuplot commands:")
    print(f"  plot '{output_file}' using 1:2 with points title 'weights'")

    # Optional sanity plots with matplotlib
    try:
        import matplotlib.pyplot as plt
        HAVE_MPL = True
    except ImportError:
        HAVE_MPL = False

    if HAVE_MPL:
        # Histogram of samples (density) vs q(x) and f(x) shape
        plt.figure()
        plt.hist(samples, bins=50, density=True, alpha=0.6, label="samples ~ q(x)")

        # Plot numerical q(x) (from grid)
        plt.plot(xs_grid, qnorm_grid, 'k-', label="q(x) numeric")

        # Plot scaled f(x) for shape comparison
        xfine = [q_a + (q_b - q_a)*i/1000.0 for i in range(1001)]
        fvals = [max(0.0, f(x)) for x in xfine]  # in case f can be negative, clip for plotting
        maxf = max(fvals) if fvals else 1.0
        if maxf == 0.0:
            maxf = 1.0
        f_scaled = [val / maxf * max(qnorm_grid) for val in fvals]
        plt.plot(xfine, f_scaled, 'r-', label="scaled f(x)")

        plt.xlabel("x")
        plt.ylabel("density / scaled f(x)")
        plt.title("Importance sampling: proposal q(x) and target shape")
        plt.legend()
        plt.grid(True)
        plt.show()

    return mean_w, std_error


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    importance_sampling()