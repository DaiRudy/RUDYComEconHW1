import numpy as np   
import matplotlib.pyplot as plt

###############################################
# 1. Tauchen Discretization of AR(1) Process
###############################################

def tauchen_discretize(N, rho, mu, sigma_u, m=3.0):
    """
    Discretize AR(1) process using Tauchen's method:
       z_{t+1} = (1-rho)*mu + rho*z_t + sigma_u * eps, where eps ~ N(0,1)
       
    Returns:
       z_grid: Array of state values of length N
       P:      Transition probability matrix of shape (N, N)
    """
    std_z = np.sqrt(sigma_u**2 / (1 - rho**2))
    z_min = mu - m * std_z
    z_max = mu + m * std_z
    
    z_grid = np.linspace(z_min, z_max, N)
    step = (z_max - z_min) / (N - 1)
    
    P = np.zeros((N, N))
    from math import erf, sqrt
    def normal_cdf(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    
    for i in range(N):
        z_i = z_grid[i]
        mean_i = (1 - rho)*mu + rho*z_i
        for j in range(N):
            if j == 0:
                z_left  = z_grid[0] - step/2
                z_right = z_grid[0] + step/2
            elif j == N-1:
                z_left  = z_grid[N-1] - step/2
                z_right = z_grid[N-1] + step/2
            else:
                z_left  = z_grid[j] - step/2
                z_right = z_grid[j] + step/2
            
            z_left_std  = (z_left  - mean_i) / sigma_u
            z_right_std = (z_right - mean_i) / sigma_u
            P[i, j] = normal_cdf(z_right_std) - normal_cdf(z_left_std)
    
    return z_grid, P


###############################################
# 2. Construct (w, z) -> (w', z') Transition Matrix
###############################################

def build_transition_matrix(w_grid, z_grid, Pz, 
                            w_hat=1.0, s_0=0.75,
                            c_y=1.0, mu_y=1.0,
                            c_r=0.05, mu_r=0.1):
    """
    Construct transition matrix P_wz on the (w, z) discrete state space.
    The wealth update equation is:
       w' = R(z) * s(w) + y(z)
       
    R(z) = c_r * exp(z) + exp(mu_r)
    y(z) = c_y * exp(z) + exp(mu_y)
    s(w) = s_0 * w (if w >= w_hat, else 0)
    """
    Nw = len(w_grid)
    Nz = len(z_grid)
    P = np.zeros((Nw * Nz, Nw * Nz))
    
    def R_of_z(z):
        return c_r * np.exp(z) + np.exp(mu_r)
    
    def y_of_z(z):
        return c_y * np.exp(z) + np.exp(mu_y)
    
    def s_of_w(w):
        return s_0 * w if w >= w_hat else 0.0
    
    # For each (w, z) state, first compute the next period's continuous wealth w_next_cont,
    # then map this continuous value to the wealth grid w_grid using linear interpolation (giving weights to adjacent discrete points).
    for iz in range(Nz):
        z_now = z_grid[iz]
        for iw in range(Nw):
            w_now = w_grid[iw]
            w_next_cont = R_of_z(z_now) * s_of_w(w_now) + y_of_z(z_now)
            
            # Interpolate using the wealth grid
            if w_next_cont <= w_grid[0]:
                w_next_probs = np.zeros(Nw)
                w_next_probs[0] = 1.0
            elif w_next_cont >= w_grid[-1]:
                w_next_probs = np.zeros(Nw)
                w_next_probs[-1] = 1.0
            else:
                j = np.searchsorted(w_grid, w_next_cont) - 1
                j = max(min(j, Nw-2), 0)
                w_left, w_right = w_grid[j], w_grid[j+1]
                alpha = (w_next_cont - w_left) / (w_right - w_left)
                w_next_probs = np.zeros(Nw)
                w_next_probs[j]   = 1 - alpha
                w_next_probs[j+1] = alpha
            
            from_idx = iz * Nw + iw
            # For the z part, transition according to Pz
            for iz_next in range(len(z_grid)):
                p_z = Pz[iz, iz_next]
                if p_z > 1e-14:
                    to_base = iz_next * Nw
                    for iw_next in range(Nw):
                        p_w = w_next_probs[iw_next]
                        if p_w > 0:
                            to_idx = to_base + iw_next
                            P[from_idx, to_idx] += p_z * p_w
    return P


###############################################
# 3. Compute Stationary Distribution
###############################################

def stationary_dist(P, pi0=None, max_iter=1000, tol=1e-12):
    """
    Given transition matrix P, compute the stationary distribution satisfying pi = pi * P.
    If pi0 is None, use a uniform distribution as the initial distribution.
    """
    S = P.shape[0]
    if pi0 is None:
        pi = np.ones(S) / S
    else:
        pi = pi0.copy()
    
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.max(np.abs(pi_next - pi)) < tol:
            return pi_next
        pi = pi_next
    return pi


###############################################
# 4. Construct Initial Distribution: Only Assign Positive Probability to w in [w_low, w_high] on the Wealth Grid
###############################################

def build_initial_distribution(w_grid, z_grid, w_low=1.0, w_high=10.0):
    """
    Construct initial distribution pi0 (shape (Nw * Nz,)):
      - Only assign positive probability to discrete wealth w in the interval [w_low, w_high];
      - Use a uniform distribution in the z direction.
      
    Returns:
       pi0: Initial distribution vector, where all positive probability states correspond to discrete points on w_grid.
    """
    Nw = len(w_grid)
    Nz = len(z_grid)
    
    # Find indices on the wealth grid that satisfy the condition
    valid_w_indices = np.where((w_grid >= w_low) & (w_grid <= w_high))[0]
    if valid_w_indices.size == 0:
        raise ValueError("No wealth grid points fall within [w_low, w_high], please check parameter settings.")
    
    # Assign positive probability only to these discrete w points across all z states
    pi0 = np.zeros(Nw * Nz)
    num_valid = valid_w_indices.size * Nz  # Total number of states to assign positive probability
    mass = 1.0 / num_valid  # Equal probability mass
    
    for iz in range(Nz):
        for iw in valid_w_indices:
            idx = iz * Nw + iw
            pi0[idx] = mass
            
    return pi0


###############################################
# 5. Main Program Example
###############################################

if __name__ == "__main__":
    # ------------------ (1) Parameter Settings ------------------
    a = 0.5
    b = 0.0
    sigma_z = 0.1
    rho = a
    mu = b / (1 - a)
    sigma_u = sigma_z
    
    # Tauchen discretization of z: Select 5 states
    Nz = 5
    z_grid, Pz = tauchen_discretize(N=Nz, rho=rho, mu=mu, sigma_u=sigma_u, m=3.0)
    
    # Generate discrete wealth grid: e.g., 0, 4, 8, …, 200
    w_grid = np.arange(0, 201, 4)
    
    # Other wealth parameters
    w_hat = 1.0
    s_0 = 0.75
    c_y, mu_y = 1.0, 1.0
    c_r, mu_r = 0.05, 0.1
    
    # ------------------ (2) Construct (w,z) -> (w',z') Transition Matrix ------------------
    P_wz = build_transition_matrix(w_grid, z_grid, Pz, 
                                   w_hat=w_hat, s_0=s_0,
                                   c_y=c_y,   mu_y=mu_y,
                                   c_r=c_r,   mu_r=mu_r)
    
    # ------------------ (3) Construct Initial Distribution and Compute Stationary Distribution ------------------
    # Here, the initial distribution assigns positive probability only to discrete points on w_grid within [1.0, 10.0],
    # so the initial wealth w0 can only take these discrete values (e.g., if w_grid = [0,4,8,12,...], valid points might be 4 and 8)
    pi0 = build_initial_distribution(w_grid, z_grid, w_low=1.0, w_high=10.0)
    
    # Compute stationary distribution
    pi_wz = stationary_dist(P_wz, pi0=pi0)
    
    # Convert 1D stationary distribution to (z, w) shape
    pi_wz_mat = pi_wz.reshape((len(z_grid), len(w_grid)))
    
    # ------------------ (4) Plot Results ------------------
    # (a) Plot marginal distribution of w (sum over z)
    p_w = pi_wz_mat.sum(axis=0)
    plt.figure(figsize=(8,5))
    plt.bar(w_grid, p_w, width=(w_grid[1]-w_grid[0])*0.9, alpha=0.6)
    plt.xlabel('Wealth (w)')
    plt.ylabel('Probability')
    plt.title('Stationary Distribution of Wealth')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # (b) Plot (w, z) joint distribution heatmap
    plt.figure(figsize=(8,6))
    plt.imshow(pi_wz_mat, origin='lower', aspect='auto',
               extent=[w_grid[0], w_grid[-1], z_grid[0], z_grid[-1]])
    plt.colorbar(label='Probability')
    plt.xlabel('Wealth (w)')
    plt.ylabel('z')
    plt.title('Joint Distribution p(w, z)')
    plt.show()
    
    ###############################################
    # (E) New Section: Replot (w, z) Joint Distribution Heatmap After Removing Top 1% Wealth
    ###############################################
    S = len(pi_wz)
    sample_size = 10000  
    indices = np.arange(S)
    p_normalized = pi_wz / np.sum(pi_wz)
    sampled_indices = np.random.choice(indices, size=sample_size, p=p_normalized)
    
    # Convert sample indices to corresponding z and w grid indices
    iz_samples = sampled_indices // len(w_grid)
    iw_samples = sampled_indices % len(w_grid)
    w_samples = w_grid[iw_samples]
    z_samples = z_grid[iz_samples]
    
    # Compute 99th percentile of w (remove top 1%)
    w_threshold = np.percentile(w_samples, 99)
    mask = w_samples < w_threshold
    w_filtered = w_samples[mask]
    z_filtered = z_samples[mask]
    
    num_w_bins = 50
    num_z_bins = 50
    H_filtered, w_edges_filtered, z_edges_filtered = np.histogram2d(
        w_filtered, z_filtered,
        bins=[num_w_bins, num_z_bins],
        range=[[w_grid[0], w_threshold],
               [z_grid[0], z_grid[-1]]],
        density=True
    )
    
    plt.figure(figsize=(8,6))
    plt.imshow(H_filtered.T, origin='lower', aspect='auto',
               extent=[w_grid[0], w_threshold, z_grid[0], z_grid[-1]],
               cmap='viridis')
    plt.colorbar(label="Density")
    plt.xlabel("Wealth (w)")
    plt.ylabel("z")
    plt.title("Joint Distribution (w, z) with Top 1% Wealth Removed")
    plt.show()