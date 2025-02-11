import numpy as np
import matplotlib.pyplot as plt

###############################################
# 1. Auxiliary function for discretizing an AR(1) process using Tauchen's method
###############################################

def tauchen_discretize(N, rho, mu, sigma_u, m=3.0):
    """
    Discretize the AR(1) process using Tauchen's method:
        z_{t+1} = (1 - rho) * mu + rho * z_t + sigma_u * eps, where eps ~ N(0,1)
    Returns:
        z_grid: array of shape (N,) containing the discretized state values for z
        Pz:     transition probability matrix of shape (N, N)
    """
    std_z = np.sqrt(sigma_u**2 / (1 - rho**2))
    z_min = mu - m * std_z
    z_max = mu + m * std_z

    z_grid = np.linspace(z_min, z_max, N)
    step = (z_max - z_min) / (N - 1)

    P = np.zeros((N, N))
    from math import erf
    def normal_cdf(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    for i in range(N):
        z_i = z_grid[i]
        mean_i = (1 - rho) * mu + rho * z_i
        for j in range(N):
            if j == 0:
                z_left  = z_grid[0] - step / 2
                z_right = z_grid[0] + step / 2
            elif j == N - 1:
                z_left  = z_grid[N - 1] - step / 2
                z_right = z_grid[N - 1] + step / 2
            else:
                z_left  = z_grid[j] - step / 2
                z_right = z_grid[j] + step / 2

            z_left_std  = (z_left - mean_i) / sigma_u
            z_right_std = (z_right - mean_i) / sigma_u
            P[i, j] = normal_cdf(z_right_std) - normal_cdf(z_left_std)

    return z_grid, P

###############################################
# 2. Compute the stationary distribution for a given Markov chain P
###############################################

def stationary_dist(P, max_iter=1000, tol=1e-12):
    """
    Iteratively solve for the stationary distribution pi such that pi = pi * P,
    where P is the transition matrix of shape (S, S).
    """
    S = P.shape[0]
    pi = np.ones(S) / S
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.max(np.abs(pi_next - pi)) < tol:
            return pi_next
        pi = pi_next
    return pi

###############################################
# 3. Model functions: R(z), y(z), and s(w)
###############################################

def R_of_z(z, c_r=0.05, mu_r=0.1):
    """
    R(z) = c_r * exp(z) + exp(mu_r).
    (If the model requires (1+R), add an extra 1 separately.)
    """
    return c_r * np.exp(z) + np.exp(mu_r)

def y_of_z(z, c_y=1.0, mu_y=1.0):
    """
    y(z) = c_y * exp(z) + exp(mu_y)
    """
    return c_y * np.exp(z) + np.exp(mu_y)

def s_of_w(w, w_hat=1.0, s_0=0.75):
    """
    Savings function: s(w) = s_0 * w if w >= w_hat; otherwise, 0.
    """
    return s_0 * w if w >= w_hat else 0.0

###############################################
# 4. Construct a "non-stochastic" sequence: z_seq[0..T-1]
###############################################
#   Objective: Ensure that over T periods, each state i appears roughly pi[i]*T times,
#              and the transitions from i to j roughly match P[i, j].
#   This is a simplified demonstration version; in practice, a more systematic Eulerian path algorithm can be used.
###############################################

def build_nonstochastic_sequence(P, pi, T):
    """
    Build a state sequence z_seq of length T, such that the long-run frequencies approximate P and pi.

    Parameters
    ----------
    P  : Transition matrix of shape (Nz, Nz)
    pi : Stationary distribution of shape (Nz,)
    T  : Integer, length of the sequence

    Returns
    -------
    z_seq : ndarray of shape (T,), with values in 0 .. (Nz - 1)
    """
    Nz = len(pi)

    # Step A: Allocate the number of occurrences for each state i
    visits = np.round(pi * T).astype(int)
    diff = T - visits.sum()
    # If the rounded totals do not sum to T, adjust slightly
    while diff != 0:
        if diff > 0:
            i_add = np.random.randint(Nz)
            visits[i_add] += 1
            diff -= 1
        else:  # diff < 0
            i_sub = np.random.randint(Nz)
            if visits[i_sub] > 0:
                visits[i_sub] -= 1
                diff += 1

    # Step B: Allocate the number of transitions from state i to state j based on P[i, j]
    trans = np.zeros((Nz, Nz), dtype=int)
    for i in range(Nz):
        if visits[i] == 0:
            continue
        row_float = P[i, :] * visits[i]
        row_int   = np.round(row_float).astype(int)
        row_diff  = visits[i] - row_int.sum()
        while row_diff != 0:
            if row_diff > 0:
                j_add = np.random.randint(Nz)
                row_int[j_add] += 1
                row_diff -= 1
            else:
                j_sub = np.random.randint(Nz)
                if row_int[j_sub] > 0:
                    row_int[j_sub] -= 1
                    row_diff += 1
        trans[i, :] = row_int

    # May need further row and column adjustments; here we simplify by ensuring each row sums to visits[i]
    for i in range(Nz):
        outdeg = trans[i, :].sum()
        if outdeg < visits[i]:
            # Add to the most probable j
            missing = visits[i] - outdeg
            j_star = np.argmax(P[i, :])
            trans[i, j_star] += missing
        elif outdeg > visits[i]:
            # Subtract the excess
            extra = outdeg - visits[i]
            j_star = np.argmax(trans[i, :])
            trans[i, j_star] -= extra

    # Step C: (Simplified) Find a path from the graph
    graph = trans.copy()
    start = np.argmax(visits)
    z_seq = [start]
    cur = start
    for _ in range(T - 1):
        row = graph[cur, :]
        if row.sum() == 0:
            # If there are no remaining edges from this state, randomly choose another state with available transitions
            candidates = np.where(graph.sum(axis=1) > 0)[0]
            if len(candidates) > 0:
                cur = np.random.choice(candidates)
                z_seq.append(cur)
                continue
            else:
                break
        # Choose the edge with the largest remaining count
        j_star = np.argmax(row)
        graph[cur, j_star] -= 1
        cur = j_star
        z_seq.append(cur)
    return np.array(z_seq, dtype=int)

###############################################
# 5. Simulate the evolution of wealth w_t over time using the non-stochastic path
###############################################

def simulate_wealth_nonstochastic(z_seq, z_grid, w0=2.0, 
                                  T=None,
                                  w_hat=1.0, s_0=0.75,
                                  c_y=1.0, mu_y=1.0,
                                  c_r=0.05, mu_r=0.1):
    """
    Given a "non-stochastic" z sequence and an initial wealth w0, iterate over time:
        w_{t+1} = (R(z_t)) * s(w_t) + y(z_t)
    (Note: If the model requires (1+R(z_t)), modify R_of_z accordingly.)

    Returns:
        w_path: ndarray of shape (T+1,)
    """
    if T is None:
        T = len(z_seq)
    w_path = np.zeros(T + 1)
    w_path[0] = w0

    for t in range(T):
        idx_z = z_seq[t]        # z_seq contains discrete state indices
        z_val = z_grid[idx_z]     # Get the actual float value of z

        wt = w_path[t]
        Rt = R_of_z(z_val, c_r=c_r, mu_r=mu_r)
        yt = y_of_z(z_val, c_y=c_y, mu_y=mu_y)
        st = s_of_w(wt, w_hat=w_hat, s_0=s_0)
        w_path[t + 1] = Rt * st + yt

    return w_path

###############################################
# 6. Main program: Simulate multiple runs for multiple initial wealth values,
#    then plot the distribution of final wealth values and the joint (w, z) distribution.
###############################################

if __name__ == "__main__":
    # ---------- (1) Discretize z using Tauchen's method ----------
    a = 0.5
    b = 0.0
    sigma_z = 0.1

    rho = a
    mu = b / (1 - a)
    # Use 11 states for z
    z_grid, Pz = tauchen_discretize(N=11, rho=rho, mu=mu, sigma_u=sigma_z, m=3.0)

    # (Optional) Compute the stationary distribution for z
    pi_z = stationary_dist(Pz)

    # ---------- (2) Simulation parameters ----------
    T_sim = 50  # Simulation horizon (number of periods per simulation)
    # Create 201 initial wealth values from 0 to 200 (inclusive)
    w0_grid = np.linspace(0, 200, 201)  # 201 points
    runs_per_initial = 50             # 50 simulation runs for each initial wealth value

    # Lists to store the final wealth and final z from each simulation run
    final_wealths = []
    final_zs = []

    # ---------- (3) Run simulations ----------
    # For each initial wealth value, perform 'runs_per_initial' simulations
    for w0 in w0_grid:
        for run in range(runs_per_initial):
            # Generate a non-stochastic shock sequence for T_sim periods
            z_seq = build_nonstochastic_sequence(Pz, pi_z, T_sim)
            # Simulate the wealth evolution for the given initial wealth
            w_path = simulate_wealth_nonstochastic(z_seq, z_grid, w0=w0, T=T_sim,
                                                   w_hat=1.0, s_0=0.75,
                                                   c_y=1.0, mu_y=1.0,
                                                   c_r=0.05, mu_r=0.1)
            # Record the final wealth value (after T_sim periods)
            final_w = w_path[-1]
            # Record the final z value (the last shock used)
            final_z = z_grid[z_seq[-1]]
            final_wealths.append(final_w)
            final_zs.append(final_z)

    # Convert lists to numpy arrays for easier processing
    final_wealths = np.array(final_wealths)  # Shape: (201*50,) = (10050,)
    final_zs = np.array(final_zs)

    # ---------- (4) Discretize final wealth ----------
    # It is stipulated that the final wealth can only be 0, 4, 8, ...; 
    # if the wealth value obtained by a simulation is not at these points, it will be distributed according to the linear weight based on the distance
    w_min = 0
    w_max = np.ceil(final_wealths.max() / 4) * 4
    w_grid_discrete = np.arange(w_min, w_max + 4, 4)
    w_distribution = np.zeros(len(w_grid_discrete))

    for w in final_wealths:
        # If the simulation result is less than the minimum discrete point or greater than the maximum discrete point, it is directly classified as a boundary
        if w <= w_grid_discrete[0]:
            w_distribution[0] += 1
        elif w >= w_grid_discrete[-1]:
            w_distribution[-1] += 1
        else:
            # Find w between [lower, upper]
            idx = np.searchsorted(w_grid_discrete, w) - 1
            lower = w_grid_discrete[idx]
            upper = w_grid_discrete[idx + 1]
            # Linear interpolation weights
            weight_lower = (upper - w) / (upper - lower)
            weight_upper = (w - lower) / (upper - lower)
            w_distribution[idx] += weight_lower
            w_distribution[idx + 1] += weight_upper

    # Normalized to get probability distribution
    w_distribution /= len(final_wealths)

    # ---------- (5) Plot the discrete distribution of final wealth ----------
    plt.figure(figsize=(8, 6))
    plt.bar(w_grid_discrete, w_distribution, width=3, align='center', edgecolor='k', alpha=0.7)
    plt.xlabel("Discrete Final Wealth")
    plt.ylabel("Probability")
    plt.title("Discrete Distribution of Final Wealth")
    plt.grid(True, alpha=0.3)
    plt.show()

# ---------- (6) Calculate the joint discrete distribution of final wealth and final z ----------
# Construct a two-dimensional array, with rows corresponding to discrete wealth (w_grid_discrete) and columns corresponding to discrete z (z_grid)
    joint_distribution = np.zeros((len(w_grid_discrete), len(z_grid)))

    for i in range(len(final_wealths)):
        w = final_wealths[i]
        z = final_zs[i]
        # z comes from z_grid (after discretization), find the corresponding index
        z_index = np.argmin(np.abs(z_grid - z))
        if w <= w_grid_discrete[0]:
            joint_distribution[0, z_index] += 1
        elif w >= w_grid_discrete[-1]:
            joint_distribution[-1, z_index] += 1
        else:
            idx = np.searchsorted(w_grid_discrete, w) - 1
            lower = w_grid_discrete[idx]
            upper = w_grid_discrete[idx+1]
            weight_lower = (upper - w) / (upper - lower)
            weight_upper = (w - lower) / (upper - lower)
            joint_distribution[idx, z_index] += weight_lower
            joint_distribution[idx+1, z_index] += weight_upper

    joint_distribution /= len(final_wealths)

# ---------- (7) Draw joint discrete distribution ----------
# For intuitive display, we use imshow to draw a two-dimensional heat map. Note that imshow requires that the matrix rows correspond to the y-axis,
# Therefore, we transpose joint_distribution here so that the x-axis is the wealth discrete points and the y-axis is the z discrete points.
    plt.figure(figsize=(8, 6))
    plt.imshow(joint_distribution.T, origin='lower', aspect='auto',
               extent=[w_grid_discrete[0]-2, w_grid_discrete[-1]+2, z_grid[0], z_grid[-1]],
               cmap='viridis')
    plt.colorbar(label="Probability")
    plt.xlabel("Final Wealth (Discrete)")
    plt.ylabel("Final z")
    plt.title("Joint Discrete Distribution of Final Wealth and z")
    plt.xticks(w_grid_discrete)
    plt.yticks(np.round(z_grid, 2))
    plt.show()


    