"""
Unified finite-difference solver for general linear second-order ODEs.

Equation form:
    y''(x) + P(x)*y'(x) + Q(x)*y(x) = R(x),   x in [a, b]

Supported boundary/initial (derivative) conditions:
    1) Pure BVP:
           y(a) = A,  y(b) = B
    2) Pure "IVP-style":
           y(a) = A,  y'(a) = D
    3) Mixed:
           y(a) = A,  y'(b) = D
           y'(a) = D, y(b) = B
    4) Any combination of "BC" (value) and "IC" (derivative) at each end.

Discretization:
    On a uniform grid x_i = a + i*h, i = 0..N, h = (b-a)/N.

    Approximations:
        y''(x_i) ≈ (y_{i+1} - 2y_i + y_{i-1}) / h^2
        y'(x_i)  ≈ (y_{i+1} - y_{i-1}) / (2h)

    Plugged into:
        y'' + P_i * y' + Q_i * y = R_i

    gives the interior equation:
        A_{i,i-1} * y_{i-1} + A_{i,i} * y_i + A_{i,i+1} * y_{i+1} = R_i

    with:
        A_{i,i-1} =  1/h^2 - P_i/(2h)
        A_{i,i}   = -2/h^2 + Q_i
        A_{i,i+1} =  1/h^2 + P_i/(2h)

Boundary / derivative conditions:
    - "BC" at left:   y(a)  = left_value
    - "IC" at left:   y'(a) = left_value  ≈ (y_1 - y_0)/h
    - "BC" at right:  y(b)  = right_value
    - "IC" at right:  y'(b) = right_value ≈ (y_N - y_{N-1})/h

Output:
    - Writes x, y_numeric to 'finite_difference_flexible.dat'
    - Prints example gnuplot command
    - If matplotlib is available, plots y(x) for sanity check.

How to use in exam:
-------------------
1) Set a, b, N in USER PARAMETERS.
2) Choose left_type, right_type as "BC" or "IC" and set left_value, right_value.
3) Define functions P(x), Q(x), R(x) for your ODE:
       y'' + P(x)*y' + Q(x)*y = R(x).
   Examples:
       - y'' + y = 0:
             P(x) = 0, Q(x) = 1, R(x) = 0
       - y'' = x:
             P(x) = 0, Q(x) = 0, R(x) = x
4) Run:
       python3 finite_difference_BVP_IVP_general.py
5) Plot in gnuplot:
       plot 'finite_difference_flexible.dat' using 1:2 with linespoints title 'y(x)'
"""

import math

# =============================================================
# USER PARAMETERS
# =============================================================

a = 0.0
b = 1.0
N = 40                          # number of intervals → N+1 grid points
h = (b - a) / N

# Choose condition types: "BC" or "IC"
# BC = boundary condition for y
# IC = derivative condition for y' (initial-type)
left_type  = "BC"               # "BC" or "IC"
right_type = "IC"               # "BC" or "IC"

# Values:
# If left_type  == "BC":  y(a)  = left_value
# If left_type  == "IC":  y'(a) = left_value
left_value  = 1.0

# If right_type == "BC":  y(b)  = right_value
# If right_type == "IC":  y'(b) = right_value
right_value = 0.0

output_file = "finite_difference_flexible.dat"

# =============================================================
# ODE COEFFICIENT FUNCTIONS: y'' + P(x)*y' + Q(x)*y = R(x)
# =============================================================

def P(x):
    """
    Coefficient of y'(x) in the ODE.
    For y'' + y = 0, P(x) = 0.
    """
    return 0.0


def Q(x):
    """
    Coefficient of y(x) in the ODE.
    For y'' + y = 0, Q(x) = 1.
    """
    return 1.0


def R(x):
    """
    Right-hand side R(x) in the ODE:
        y'' + P(x)*y' + Q(x)*y = R(x)
    For y'' + y = 0, R(x) = 0.
    """
    return 0.0


# =============================================================
# Build FD system A y = b
# =============================================================

def build_system():
    """
    Construct the (N+1) x (N+1) linear system for the FD scheme.

    Interior nodes (i = 1..N-1):
        A_{i,i-1} =  1/h^2 - P_i/(2h)
        A_{i,i}   = -2/h^2 + Q_i
        A_{i,i+1} =  1/h^2 + P_i/(2h)
        b_i       =  R_i

    Boundaries:
        Left:
            if left_type == "BC":  y_0 = left_value
            if left_type == "IC":  (y_1 - y_0)/h = left_value
        Right:
            if right_type == "BC": y_N = right_value
            if right_type == "IC": (y_N - y_{N-1})/h = right_value
    """
    # NOTE: This uses numpy only for convenience. In the exam, you can
    #       replace this with your own Gaussian elimination with pivoting.
    import numpy as np

    size = N + 1
    A = np.zeros((size, size), float)
    b_vec = np.zeros(size, float)

    h2 = h * h

    # ----------------- interior nodes -----------------
    for i in range(1, N):
        x_i = a + i * h
        P_i = P(x_i)
        Q_i = Q(x_i)
        R_i = R(x_i)

        A[i, i-1] =  1.0 / h2 - P_i / (2.0 * h)
        A[i, i]   = -2.0 / h2 + Q_i
        A[i, i+1] =  1.0 / h2 + P_i / (2.0 * h)
        b_vec[i]  =  R_i

    # ----------------- left boundary -----------------
    if left_type == "BC":
        # y(a) = left_value  →  y_0 = left_value
        A[0, 0] = 1.0
        b_vec[0] = left_value

    elif left_type == "IC":
        # y'(a) = left_value ≈ (y_1 - y_0) / h
        # => -1/h * y_0 + 1/h * y_1 = left_value
        A[0, 0] = -1.0 / h
        A[0, 1] =  1.0 / h
        b_vec[0] = left_value

    else:
        raise ValueError("left_type must be 'BC' or 'IC'")

    # ----------------- right boundary -----------------
    if right_type == "BC":
        # y(b) = right_value → y_N = right_value
        A[N, N] = 1.0
        b_vec[N] = right_value

    elif right_type == "IC":
        # y'(b) = right_value ≈ (y_N - y_{N-1}) / h
        # => -1/h * y_{N-1} + 1/h * y_N = right_value
        A[N, N-1] = -1.0 / h
        A[N, N]   =  1.0 / h
        b_vec[N]  = right_value

    else:
        raise ValueError("right_type must be 'BC' or 'IC'")

    return A, b_vec


# =============================================================
# Solve system and output
# =============================================================

def solve_fd():
    import numpy as np
    A, b_vec = build_system()

    # Solve A y = b_vec
    y = np.linalg.solve(A, b_vec)

    # Write output to .dat file
    with open(output_file, "w") as fout:
        fout.write("# x, y_numeric\n")
        for i in range(N + 1):
            x_i = a + i * h
            fout.write(f"{x_i} {y[i]}\n")

    print("FD solution written to:", output_file)
    print("Example gnuplot command:")
    print(f"  plot '{output_file}' using 1:2 with linespoints title 'y(x)'")

    # Optional matplotlib plot of y(x) for sanity check
    try:
        import matplotlib.pyplot as plt
        HAVE_MPL = True
    except ImportError:
        HAVE_MPL = False

    if HAVE_MPL:
        xs = [a + i * h for i in range(N + 1)]
        plt.figure()
        plt.plot(xs, y, marker='o')
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.title("Finite Difference Solution y(x)")
        plt.grid(True)
        plt.show()

    return y


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    y = solve_fd()
    print("Final y(b) =", y[-1])