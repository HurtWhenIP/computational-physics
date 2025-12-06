"""
Shooting method using RK4 for a second-order BVP:

    y'' = f(x, y, y'),  x in [a, b]
    y(a) = alpha
    y(b) = beta

Basic idea:
- Rewrite as a first-order system:
      y1 = y
      y2 = y'
      y1' = y2
      y2' = f(x, y1, y2)
- Guess the initial slope s = y'(a).
- Solve the IVP:
      y1(a) = alpha,  y2(a) = s
  up to x = b using RK4.
- Define:
      F(s) = y1(b; s) - beta
  and adjust s so that F(s) ≈ 0.
- Here we use a simple secant method on s.

What this script does:
- Allows you to specify:
    * f(x, y, y') in the system.
    * a, b, alpha, beta.
- Uses two initial guesses s0, s1 for the slope.
- Uses secant updates on s until |F(s)| < slope_tol or max_iter is reached.
- Writes:
    * shooting_convergence.dat  : s_k and F(s_k)
    * shooting_solution.dat     : x, y(x) for the final slope.
- Optional matplotlib plots of solution and shooting convergence.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * a, b, alpha, beta
    * s0, s1  (initial slope guesses)
    * N_steps (for each IVP integration)
- Step 2: Replace f(x, y, yp) with the given ODE's right-hand side.
- Step 3: Run:
        python3 shooting_method_rk4.py
- Step 4: gnuplot:
        plot 'shooting_solution.dat' using 1:2 with linespoints title 'y(x)'
        plot 'shooting_convergence.dat' using 1:3 with linespoints title 'F(s)'

Data files:
    shooting_convergence.dat:
        col1: k         (iteration index for slope)
        col2: s_k       (slope guess)
        col3: F(s_k)    (mismatch at x = b)
    shooting_solution.dat:
        col1: x
        col2: y(x) for final slope
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
a = 0.0          # left boundary
b = 1.0          # right boundary
alpha = 0.0      # y(a)
beta = 1.0       # y(b)

# Initial guesses for slope y'(a)
s0 = 0.0
s1 = 2.0

N_steps = 100             # steps for each IVP integration
slope_tol = 1e-6          # tolerance on |F(s)|
max_iter = 20             # max slope iterations

output_shoot = "shooting_convergence.dat"
output_sol = "shooting_solution.dat"
# ===============================================================


def f_second_order(x, y, yp):
    """
    Define y'' = f(x, y, y') here.

    Example:
        y'' = 2  =>  y is quadratic; BVP can be set appropriately.
    In general, replace with the given ODE.
    """
    return 2.0


def rk4_step_system(x, y1, y2, h):
    """
    Single RK4 step for the system:
        y1' = y2
        y2' = f_second_order(x, y1, y2)
    """
    # k's for y1 and y2
    k1_1 = y2
    k1_2 = f_second_order(x, y1, y2)

    k2_1 = y2 + 0.5 * h * k1_2
    k2_2 = f_second_order(x + 0.5 * h, y1 + 0.5 * h * k1_1, y2 + 0.5 * h * k1_2)

    k3_1 = y2 + 0.5 * h * k2_2
    k3_2 = f_second_order(x + 0.5 * h, y1 + 0.5 * h * k2_1, y2 + 0.5 * h * k2_2)

    k4_1 = y2 + h * k3_2
    k4_2 = f_second_order(x + h, y1 + h * k3_1, y2 + h * k3_2)

    y1_new = y1 + (h / 6.0) * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
    y2_new = y2 + (h / 6.0) * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
    return y1_new, y2_new


def solve_ivp_for_slope(s):
    """
    Given a slope guess s = y'(a), solve the IVP:
        y1(a) = alpha,  y2(a) = s
    from x = a to x = b with RK4 using N_steps.

    Returns:
        xs : list of x values
        ys : list of y values (y1)
    """
    h = (b - a) / float(N_steps)
    x = a
    y1 = alpha
    y2 = s

    xs = [x]
    ys = [y1]

    for _ in range(N_steps):
        y1, y2 = rk4_step_system(x, y1, y2, h)
        x += h
        xs.append(x)
        ys.append(y1)

    return xs, ys


def shooting_method_secant():
    """
    Perform shooting method with secant iteration on the slope.

    Uses s0, s1 as initial guesses.
    Writes convergence data to output_shoot.
    """
    fout = open(output_shoot, "w")
    fout.write("# Shooting method: secant on slope s\n")
    fout.write("# col1: k (iteration)\n")
    fout.write("# col2: s_k\n")
    fout.write("# col3: F(s_k) = y(b; s_k) - beta\n")

    # First slope guess
    xs, ys = solve_ivp_for_slope(s0)
    F0 = ys[-1] - beta

    fout.write(f"0 {s0} {F0}\n")

    # Second slope guess
    xs, ys = solve_ivp_for_slope(s1)
    F1 = ys[-1] - beta
    fout.write(f"1 {s1} {F1}\n")

    s_prev = s0
    F_prev = F0
    s_curr = s1
    F_curr = F1

    for k in range(2, max_iter + 1):
        if abs(F_curr - F_prev) < 1e-14:
            print("Warning: F_curr - F_prev is too small, stopping secant iterations.")
            break

        # Secant update
        s_next = s_curr - F_curr * (s_curr - s_prev) / (F_curr - F_prev)

        xs, ys = solve_ivp_for_slope(s_next)
        F_next = ys[-1] - beta

        fout.write(f"{k} {s_next} {F_next}\n")

        if abs(F_next) < slope_tol:
            # Good enough
            s_curr = s_next
            xs_final = xs
            ys_final = ys
            print(f"Shooting converged at iteration {k} with slope s = {s_curr:.6f}")
            print(f"Boundary mismatch F(s) = {F_next:.6e}")
            fout.close()
            return s_curr, xs_final, ys_final

        # Prepare next secant step
        s_prev, F_prev = s_curr, F_curr
        s_curr, F_curr = s_next, F_next

    # If we exit loop without early return, use last result
    xs_final = xs
    ys_final = ys
    print("Shooting method reached maximum iterations.")
    print(f"Last slope s = {s_curr:.6f}, F(s) = {F_curr:.6e}")
    fout.close()
    return s_curr, xs_final, ys_final


def main():
    s_final, xs, ys = shooting_method_secant()

    # Write final solution to file
    fout = open(output_sol, "w")
    fout.write("# Shooting method final solution\n")
    fout.write("# col1: x\n")
    fout.write("# col2: y(x)\n")
    for x, y in zip(xs, ys):
        fout.write(f"{x} {y}\n")
    fout.close()

    print("Final shooting solution written to:", output_sol)
    print("Shooting convergence data written to:", output_shoot)

    if HAVE_MPL:
        # Plot y(x)
        plt.figure()
        plt.plot(xs, ys, marker='o')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Shooting method: final solution")

        # Plot F(s_k) vs iteration k
        ks = []
        Fs = []
        with open(output_shoot, "r") as fin:
            for line in fin:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                k = int(parts[0])
                Fk = float(parts[2])
                ks.append(k)
                Fs.append(Fk)

        plt.figure()
        plt.plot(ks, Fs, marker='o')
        plt.xlabel("iteration k")
        plt.ylabel("F(s_k) = y(b; s_k) - beta")
        plt.title("Shooting method: boundary mismatch vs iteration")
        plt.show()


if __name__ == "__main__":
    main()