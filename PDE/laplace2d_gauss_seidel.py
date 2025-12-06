"""
2D Laplace equation on a rectangle using Gauss–Seidel relaxation.

PDE:
    ∇²u = 0   on (x, y) in [0, Lx] × [0, Ly]

Discrete form (5-point stencil on interior grid points):
    u_{i,j} = (1/4) * (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1})

Boundary conditions:
- Dirichlet values specified on all four boundaries.

What this script does:
- Creates an (Nx+1) × (Ny+1) grid of points.
- Sets boundary values using four user-defined functions:
    * g_left(y), g_right(y), g_bottom(x), g_top(x)
- Initializes interior u_{i,j} (e.g., zeros).
- Performs Gauss–Seidel sweeps over interior points until:
      max change in u (over all interior points) < tol
  or max_iters is reached.
- Writes the final grid to 'laplace2d_solution.dat':
      i, j, x_i, y_j, u_{i,j}
- Writes convergence history to 'laplace2d_convergence.dat':
      iteration, max_change
- Optionally plots convergence with matplotlib.

How to use in the exam:
- Step 1: Edit USER PARAMETERS:
    * Lx, Ly, Nx, Ny
    * tol, max_iters
- Step 2: Edit boundary functions if needed.
- Step 3: Run:
        python3 laplace2d_gauss_seidel.py
- Step 4: gnuplot:
        # Surface-like pseudo-plot (example)
        splot 'laplace2d_solution.dat' using 3:4:5 with points
        # Convergence:
        plot 'laplace2d_convergence.dat' using 1:2 every ::1 with linespoints title 'max change'

Note:
- Error/convergence is purely numerical:
    E_k = max_{i,j} |u_{i,j}^{(k)} - u_{i,j}^{(k-1)}|.
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
Lx = 1.0       # domain size in x
Ly = 1.0       # domain size in y
Nx = 20        # number of intervals in x -> Nx+1 grid points
Ny = 20        # number of intervals in y -> Ny+1 grid points

tol = 1e-5     # stopping tolerance on max change per sweep
max_iters = 5000

output_grid_file = "laplace2d_solution.dat"
output_conv_file = "laplace2d_convergence.dat"
# ===============================================================


def g_left(y):
    """Boundary condition u(0, y). Edit in exam if needed."""
    return 0.0


def g_right(y):
    """Boundary condition u(Lx, y). Edit in exam if needed."""
    return 0.0


def g_bottom(x):
    """Boundary condition u(x, 0). Edit in exam if needed."""
    return 0.0


def g_top(x):
    """Boundary condition u(x, Ly). Edit in exam if needed."""
    # Example: top boundary at potential 1.0
    return 1.0


def initialize_grid():
    """
    Create grid arrays:
        u[i][j], i = 0..Nx, j = 0..Ny

    Apply Dirichlet boundary conditions on all four sides,
    and initialize interior to 0.
    """
    # allocate u as (Nx+1) × (Ny+1)
    u = [[0.0 for _ in range(Ny + 1)] for _ in range(Nx + 1)]

    dx = Lx / float(Nx)
    dy = Ly / float(Ny)

    # Apply boundary conditions
    # Left and right boundaries (x=0, x=Lx)
    for j in range(Ny + 1):
        y = j * dy
        u[0][j] = g_left(y)
        u[Nx][j] = g_right(y)

    # Bottom and top boundaries (y=0, y=Ly)
    for i in range(Nx + 1):
        x = i * dx
        u[i][0] = g_bottom(x)
        u[i][Ny] = g_top(x)

    # Interior is already zero by initialization
    return u, dx, dy


def gauss_seidel_laplace(u, dx, dy):
    """
    Perform Gauss–Seidel iterations for Laplace's equation.

    Since ∇²u = 0, the discrete scheme simplifies to:
        u_{i,j} = (1/4) * (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1})

    Returns:
        u          : converged solution
        iters_used : number of iterations
        changes    : list of max changes per iteration
    """
    changes = []
    dx2 = dx * dx
    dy2 = dy * dy

    # For uniform grid Laplace, the update formula is independent of dx, dy
    # but we keep them in case we extend to Poisson later.

    for it in range(1, max_iters + 1):
        max_change = 0.0

        # Loop over interior points i=1..Nx-1, j=1..Ny-1
        for i in range(1, Nx):
            for j in range(1, Ny):
                old = u[i][j]
                # 5-point stencil
                new_val = 0.25 * (
                    u[i + 1][j] + u[i - 1][j] +
                    u[i][j + 1] + u[i][j - 1]
                )
                u[i][j] = new_val
                change = abs(new_val - old)
                if change > max_change:
                    max_change = change

        changes.append(max_change)

        if max_change < tol:
            print(f"Gauss–Seidel converged in {it} iterations (max change = {max_change:.3e}).")
            return u, it, changes

    print(f"Reached max_iters = {max_iters} (last max change = {max_change:.3e}).")
    return u, max_iters, changes


def write_grid(u, dx, dy, filename):
    """
    Write final grid to file:
        i, j, x_i, y_j, u_ij
    """
    with open(filename, "w") as f:
        f.write("# 2D Laplace solution\n")
        f.write("# col1: i (x index)\n")
        f.write("# col2: j (y index)\n")
        f.write("# col3: x_i\n")
        f.write("# col4: y_j\n")
        f.write("# col5: u(i,j)\n")
        for i in range(Nx + 1):
            x = i * dx
            for j in range(Ny + 1):
                y = j * dy
                f.write(f"{i} {j} {x} {y} {u[i][j]}\n")


def write_convergence(changes, filename):
    """
    Write iteration vs max_change to a .dat file.
    """
    with open(filename, "w") as f:
        f.write("# Laplace Gauss-Seidel convergence\n")
        f.write("# col1: iteration\n")
        f.write("# col2: max_change\n")
        for k, ch in enumerate(changes):
            f.write(f"{k+1} {ch}\n")


def main():
    u, dx, dy = initialize_grid()
    u, iters, changes = gauss_seidel_laplace(u, dx, dy)

    write_grid(u, dx, dy, output_grid_file)
    write_convergence(changes, output_conv_file)

    print("Final grid written to:", output_grid_file)
    print("Convergence history written to:", output_conv_file)
    print("Example gnuplot commands:")
    print("  splot 'laplace2d_solution.dat' using 3:4:5 with points")
    print("  plot  'laplace2d_convergence.dat' using 1:2 every ::1 with linespoints title 'max change'")

    if HAVE_MPL:
        # Convergence plot
        it_list = list(range(1, len(changes) + 1))
        eps = 1e-16
        ch_no_zero = [c if c > 0.0 else eps for c in changes]

        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogy(it_list[0:], ch_no_zero[0:], marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Max change")
        plt.title("Laplace Gauss–Seidel: convergence")
        plt.show()


if __name__ == "__main__":
    main()