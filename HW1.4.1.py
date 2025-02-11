import numpy as np  
import matplotlib.pyplot as plt

###############################################
# 1. Auxiliary functions for Tauchen discretized AR(1) processes
###############################################

def tauchen_discretize(N, rho, mu, sigma_u, m=3.0):
    """
    Discretize the AR(1) process using Tauchen's method:
    z_{t+1} = (1-rho)*mu + rho*z_t + sigma_u * eps, eps ~ N(0,1)
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
# 2. Construct the transfer matrix from (w, z) -> (w', z')
###############################################

def build_transition_matrix(w_grid, z_grid, Pz, 
                            w_hat=1.0, s_0=0.75,
                            c_y=1.0, mu_y=1.0,
                            c_r=0.05, mu_r=0.1):
    """
    Constructs the transfer matrix P_wz on the (w,z) grid.
    """
    Nw = len(w_grid)
    Nz = len(z_grid)
    P = np.zeros((Nw*Nz, Nw*Nz))
    
    def R_of_z(z):
        # R(z) = c_r*exp(z) + exp(mu_r)
        return c_r * np.exp(z) + np.exp(mu_r)

    def y_of_z(z):
        return c_y * np.exp(z) + np.exp(mu_y)
    
    def s_of_w(w):
        return s_0 * w if w >= w_hat else 0.0
    
    for iz in range(Nz):
        z_now = z_grid[iz]
        for iw in range(Nw):
            w_now = w_grid[iw]
            
            # Next period continuous wealth
            w_next_cont = (R_of_z(z_now)) * s_of_w(w_now) + y_of_z(z_now)
            
            # Interpolate w_next_cont on w_grid
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
                alpha = (w_next_cont - w_left)/(w_right - w_left)
                w_next_probs = np.zeros(Nw)
                w_next_probs[j]   = 1 - alpha
                w_next_probs[j+1] = alpha
            
            from_idx = iz*Nw + iw
            # z' is determined by Pz
            for iz_next in range(len(z_grid)):
                p_z = Pz[iz, iz_next]
                if p_z > 1e-14:
                    to_base = iz_next*Nw
                    for iw_next in range(Nw):
                        p_w = w_next_probs[iw_next]
                        if p_w > 0:
                            to_idx = to_base + iw_next
                            P[from_idx, to_idx] += p_z * p_w
    return P


###############################################
# 3. Solve for steady-state distribution (you can specify the initial distribution pi0)
###############################################

def stationary_dist(P, pi0=None, max_iter=1000, tol=1e-12):
    """
    Given a transfer matrix P, find a steady-state distribution that satisfies pi = pi * P.
    You can specify an initial distribution pi0 (shape = (S,)).
    If pi0=None, the default is uniform distribution.
    """
    S = P.shape[0]
    if pi0 is None:
        # If not specified, uniform distribution is used.
        pi = np.ones(S) / S
    else:
        pi = pi0.copy()  # Avoid modifying original data
    
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.max(np.abs(pi_next - pi)) < tol:
            return pi_next
        pi = pi_next
    return pi


###############################################
# 4. Construct a custom "initial distribution" function
###############################################

def build_initial_distribution(w_grid, z_grid, w_low=1.0, w_high=10.0):
    """
    Construct an initial distribution pi0 (shape=(Nw*Nz,)) on the (w,z) discrete state space
    - Positive probability when w ∈ [w_low, w_high], 0 otherwise
    - Simply assign uniform probability to the z direction (can be changed as needed)
    """
    Nw = len(w_grid)
    Nz = len(z_grid)
    pi0 = np.zeros(Nw*Nz)
    
    # First count which w are in [w_low, w_high]
    valid_indices = []
    for iw, w_val in enumerate(w_grid):
        if w_low <= w_val <= w_high:
            valid_indices.append(iw)
    
    if not valid_indices:
        raise ValueError("No w grids fall within [w_low, w_high], please check the grids or intervals.")
    
    # Here we give it uniform probability in the z direction.
    for iz in range(Nz):
        for iw in valid_indices:
            idx = iz*Nw + iw
            pi0[idx] = 1.0
    
    # Normalize
    pi0 /= pi0.sum()
    return pi0


###############################################
# 5. Main program example
###############################################

if __name__ == "__main__":
    # ============ (1) Coefficients ============
    a = 0.5
    b = 0.0
    sigma_z = 0.1
    
    # Convert to Tauchen required parameters
    rho = a
    mu = b/(1 - a)  
    sigma_u = sigma_z
    
    # --- z-grid: Using Tauchen's method, select 5 states ---
    Nz = 5
    z_grid, Pz = tauchen_discretize(N=Nz, rho=rho, mu=mu, sigma_u=sigma_u, m=3.0)
    
    # --- w Grid: Generates a grid from 0 to 200 with a step size of 4 (0,4,8,...,200) ---
    w_grid = np.arange(0, 201, 4)
    
    # Other wealth parameters
    w_hat = 1.0
    s_0 = 0.75
    c_y, mu_y = 1.0, 1.0
    c_r, mu_r = 0.05, 0.1
    
    # ============ (2) Construct (w,z)->(w',z') Transfer Matrix ============
    P_wz = build_transition_matrix(w_grid, z_grid, Pz, 
                                  w_hat=w_hat, s_0=s_0,
                                  c_y=c_y,   mu_y=mu_y,
                                  c_r=c_r,   mu_r=mu_r)
    
    # ============ (3) Construct custom initial distribution & iterate to find steady state ============
    pi0 = build_initial_distribution(w_grid, z_grid, w_low=1.0, w_high=10.0)
    
    # Iterate with the specified pi0
    pi_wz = stationary_dist(P_wz, pi0=pi0)
    
    # Convert the 1D steady-state distribution to a (z,w) shape: shape = (Nz, len(w_grid))
    pi_wz_mat = pi_wz.reshape((len(z_grid), len(w_grid)))
    
    # ============ (4) Plotting ============
    # (a) The marginal distribution of w
    p_w = pi_wz_mat.sum(axis=0)  # Sum over z
    plt.figure(figsize=(8,5))
    plt.bar(w_grid, p_w, width=(w_grid[1]-w_grid[0])*0.9, alpha=0.6)
    plt.xlabel('Wealth (w)')
    plt.ylabel('Probability')
    plt.title('Stationary distribution of w')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # (b) (w,z) Heat map of the joint distribution
    plt.figure(figsize=(8,6))
    plt.imshow(pi_wz_mat, origin='lower', aspect='auto',
               extent=[w_grid[0], w_grid[-1], z_grid[0], z_grid[-1]])
    plt.colorbar(label='Probability')
    plt.xlabel('Wealth (w)')
    plt.ylabel('z')
    plt.title('Joint distribution p(w,z)')
    plt.show()
    
    ###############################################
    # (E) New part: Redraw the (w,z) joint distribution heat map after removing the largest 1% of w
    ###############################################
    # Sample from the entire (w,z) state space.
    S = len(pi_wz)
    sample_size = 10000  # Adjust as needed
    indices = np.arange(S)
    # Use np.random.choice to sample according to pi_wz (pi_wz is normalized)
    p_normalized = pi_wz / np.sum(pi_wz)
    sampled_indices = np.random.choice(indices, size=sample_size, p=p_normalized)
    
    # Convert sampled indices to corresponding z and w grid indices:
    iz_samples = sampled_indices // len(w_grid)    # z index
    iw_samples = sampled_indices % len(w_grid)       # w index
    # Map to actual values:
    w_samples = w_grid[iw_samples]
    z_samples = z_grid[iz_samples]
    
    # Calculate the 99th percentile of w (i.e. remove the top 1%)
    w_threshold = np.percentile(w_samples, 99)
    
    # Keep only samples with w less than the threshold
    mask = w_samples < w_threshold
    w_filtered = w_samples[mask]
    z_filtered = z_samples[mask]
    
    # Recalculate 2D histogram: set horizontal axis range to [w_grid[0], w_threshold]
    num_w_bins = 50
    num_z_bins = 50
    H_filtered, w_edges_filtered, z_edges_filtered = np.histogram2d(
        w_filtered, z_filtered,
        bins=[num_w_bins, num_z_bins],
        range=[[w_grid[0], w_threshold],
               [z_grid[0], z_grid[-1]]],
        density=True
    )
    
    # Draw the filtered heat map
    plt.figure(figsize=(8,6))
    plt.imshow(H_filtered.T, origin='lower', aspect='auto',
               extent=[w_grid[0], w_threshold, z_grid[0], z_grid[-1]])
    plt.colorbar(label="Density")
    plt.xlabel("Wealth (w)")
    plt.ylabel("z")
    plt.title("Joint distribution of (w,z) with top 1% wealth removed")
    plt.show()
