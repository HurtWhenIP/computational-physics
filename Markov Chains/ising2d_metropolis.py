"""
2D Ising model using Metropolis updates (single-spin flip dynamics).

Model:
    - Lx x Ly square lattice.
    - Each site (i,j) has spin s_{i,j} = +1 or -1.
    - Energy (with periodic boundaries and zero external field h = 0):
         E = -J Σ_{<i,j>} s_{i} s_{j}
      where the sum is over nearest neighbours.

Metropolis update:
    - Repeatedly pick a random site (i,j).
    - Compute the energy change ΔE if s_{i,j} -> -s_{i,j}.
    - Accept flip with probability:
          alpha = min(1, exp(-β ΔE)).
    - β = 1 / (k_B T). Here k_B is absorbed, so you set β directly.

What this script does:
    - Initializes the lattice (either ordered or random).
    - Performs N_steps single-spin Metropolis updates.
    - After each update, computes:
          M_n = (1/N_sites) Σ s_{i,j}      (magnetization per spin)
          E_n = energy per spin
      and accumulates numerical statistics on M_n:
          mean_M, var_M, lag-1 autocorr ρ1_M.
    - Tracks overall acceptance rate.

Outputs:
    - 'ising2d_series.dat'  : time series of magnetization and energy.
         col1: step index n (from 1 to N_steps)
         col2: M_n             (magnetization per spin)
         col3: E_n             (energy per spin)

    - 'ising2d_stats.dat'   : running statistics on M_n.
         col1: n
         col2: mean_M_n
         col3: var_M_n
         col4: rho1_M_n
         col5: acc_rate_n
         col6: error_mean_M_n = |mean_M_n - mean_M_prev|

How to use in the exam:
    - Edit USER PARAMETERS:
          Lx, Ly, J, beta, N_steps, init_mode.
    - Run:
          python3 ising2d_metropolis.py
    - Gnuplot examples:
          plot 'ising2d_series.dat' using 1:2 with lines title 'M'
          plot 'ising2d_series.dat' using 1:3 with lines title 'E'
"""

import math

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ========== USER PARAMETERS: CHANGE THESE IN THE EXAM ==========
Lx = 20          # lattice size in x
Ly = 20          # lattice size in y
J  = 1.0         # coupling constant
beta = 0.4       # inverse temperature (1 / T)
N_steps = 20000  # total Metropolis steps

init_mode = "random"   # "random" or "all_up"

series_file = "ising2d_series.dat"
stats_file  = "ising2d_stats.dat"

# LCG parameters
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


def create_lattice(Lx, Ly, rng, mode):
    """
    Create initial lattice of spins s = +/- 1.

    mode = "random"  : each spin is +1 or -1 with probability 1/2.
    mode = "all_up"  : all spins start at +1.
    """
    spins = []
    for i in range(Lx):
        row = []
        for j in range(Ly):
            if mode == "all_up":
                s = 1
            else:
                u = rng.rand()
                s = 1 if u < 0.5 else -1
            row.append(s)
        spins.append(row)
    return spins


def periodic_index(i, L):
    """
    Periodic boundary condition: index from 0..L-1.
    """
    if i < 0:
        return i + L
    elif i >= L:
        return i - L
    return i


def local_energy_change(spins, i, j):
    """
    Compute ΔE for flipping spin at (i,j):
        ΔE = E_new - E_old for local contribution at that site.
    Using:
        E_site = -J s_{i,j} (s_left + s_right + s_up + s_down).
    """
    Lx = len(spins)
    Ly = len(spins[0])

    s_ij = spins[i][j]

    # neighbours with periodic BC
    s_left  = spins[periodic_index(i - 1, Lx)][j]
    s_right = spins[periodic_index(i + 1, Lx)][j]
    s_down  = spins[i][periodic_index(j - 1, Ly)]
    s_up    = spins[i][periodic_index(j + 1, Ly)]

    sum_neigh = s_left + s_right + s_up + s_down

    # Old contribution: -J s_ij * sum_neigh
    # New (flipped) contribution: -J (-s_ij) * sum_neigh = +J s_ij * sum_neigh
    # So ΔE = E_new - E_old = 2 J s_ij * sum_neigh
    dE = 2.0 * J * s_ij * sum_neigh
    return dE


def total_energy(spins):
    """
    Compute total energy E for the lattice (with periodic BC).
    To avoid double-counting:
        For each site, include only interactions with right and up neighbours.
    """
    Lx = len(spins)
    Ly = len(spins[0])
    E = 0.0
    for i in range(Lx):
        for j in range(Ly):
            s_ij = spins[i][j]
            s_right = spins[periodic_index(i + 1, Lx)][j]
            s_up    = spins[i][periodic_index(j + 1, Ly)]
            E += -J * s_ij * (s_right + s_up)
    return E


def magnetization(spins):
    """
    Magnetization per spin:
        M = (1/N_sites) Σ s_{i,j}
    """
    Lx = len(spins)
    Ly = len(spins[0])
    total = 0.0
    for i in range(Lx):
        for j in range(Ly):
            total += spins[i][j]
    return total / float(Lx * Ly)


def main():
    rng = LCG(seed, A, C, M)

    spins = create_lattice(Lx, Ly, rng, init_mode)
    N_sites = Lx * Ly

    E = total_energy(spins)
    M_val = magnetization(spins)

    f_series = open(series_file, "w")
    f_series.write("# Ising 2D time series\n")
    f_series.write("# col1: step n\n")
    f_series.write("# col2: M_n (magnetization per spin)\n")
    f_series.write("# col3: E_n (energy per spin)\n")

    f_stats = open(stats_file, "w")
    f_stats.write("# Running stats on magnetization\n")
    f_stats.write("# col1: n\n")
    f_stats.write("# col2: mean_M_n\n")
    f_stats.write("# col3: var_M_n\n")
    f_stats.write("# col4: rho1_M_n (lag-1)\n")
    f_stats.write("# col5: acc_rate_n\n")
    f_stats.write("# col6: error_mean_M_n\n")

    # Running stats for magnetization M_n
    sum_M = 0.0
    sum_M2 = 0.0
    sum_MM1 = 0.0  # Σ M_n * M_{n-1}
    M_prev = None

    mean_M_prev = None
    accepted_flips = 0
    total_flips = 0

    step_list = []
    M_list = []
    E_list = []
    meanM_list = []
    varM_list = []
    rho1_list = []
    acc_list = []
    errM_list = []

    for n in range(1, N_steps + 1):
        # One Metropolis single-spin flip attempt
        total_flips += 1
        # choose random site
        u1 = rng.rand()
        u2 = rng.rand()
        i = int(u1 * Lx)
        j = int(u2 * Ly)
        if i >= Lx:
            i = Lx - 1
        if j >= Ly:
            j = Ly - 1

        dE = local_energy_change(spins, i, j)

        # Metropolis acceptance
        if dE <= 0.0:
            accept = True
        else:
            u = rng.rand()
            if u < math.exp(-beta * dE):
                accept = True
            else:
                accept = False

        if accept:
            accepted_flips += 1
            # flip spin
            spins[i][j] *= -1
            E += dE
            # update magnetization M: flip of +1->-1 changes M by -2/N_sites
            # and -1->+1 changes M by +2/N_sites
            delta_M = 2.0 * spins[i][j] / float(N_sites)   # after flip spins[i][j] is new value
            # Actually, easier: recompute magnetization occasionally, but for clarity:
            # The change from old s to new (-s) is Δs = new - old = -s - s = -2s
            # So ΔM = Δs / N_sites = -2 s_old / N_sites.
            # However we don't keep s_old here; to stay completely safe and simple in an exam,
            # we re-compute M from scratch:
            M_val = magnetization(spins)
        else:
            # no flip: E, M unchanged
            pass

        # For simplicity and clarity, recompute E per spin and M per spin
        # (this is not optimal but is straightforward in an exam).
        E_per_spin = E / float(N_sites)
        M_per_spin = M_val

        f_series.write(f"{n} {M_per_spin} {E_per_spin}\n")

        # Update running statistics of M_n
        if M_prev is not None:
            sum_MM1 += M_per_spin * M_prev
        M_prev = M_per_spin

        sum_M += M_per_spin
        sum_M2 += M_per_spin * M_per_spin

        mean_M = sum_M / float(n)
        var_M = (sum_M2 / float(n)) - mean_M * mean_M
        if n > 1 and var_M > 0.0:
            cov1 = (sum_MM1 / float(n - 1)) - mean_M * mean_M
            rho1 = cov1 / var_M
        else:
            rho1 = 0.0

        acc_rate = accepted_flips / float(total_flips)

        if mean_M_prev is None:
            err_mean = 0.0
        else:
            err_mean = abs(mean_M - mean_M_prev)

        f_stats.write(f"{n} {mean_M} {var_M} {rho1} {acc_rate} {err_mean}\n")

        step_list.append(n)
        M_list.append(M_per_spin)
        E_list.append(E_per_spin)
        meanM_list.append(mean_M)
        varM_list.append(var_M)
        rho1_list.append(rho1)
        acc_list.append(acc_rate)
        errM_list.append(err_mean)

        mean_M_prev = mean_M

    f_series.close()
    f_stats.close()

    print("Ising 2D series written to:", series_file)
    print("Stats written to:", stats_file)
    print("Final acceptance rate:", acc_list[-1] if acc_list else 0.0)

    if HAVE_MPL:
        # Magnetization time series
        plt.figure()
        plt.plot(step_list, M_list, linewidth=0.8)
        plt.xlabel("MC step")
        plt.ylabel("M_n")
        plt.title("Ising 2D: magnetization per spin")

        # Energy time series
        plt.figure()
        plt.plot(step_list, E_list, linewidth=0.8)
        plt.xlabel("MC step")
        plt.ylabel("E_n")
        plt.title("Ising 2D: energy per spin")

        # Numerical error of mean magnetization
        if len(step_list) > 1:
            eps = 1e-16
            err_no_zero = [e if e > 0.0 else eps for e in errM_list]
            plt.figure()
            plt.semilogy(step_list[2:], err_no_zero[2:], marker='o')
            plt.xlabel("MC step")
            plt.ylabel("|mean_M_n - mean_M_prev|")
            plt.title("Ising 2D: numerical convergence of <M>")

        plt.show()


if __name__ == "__main__":
    main()