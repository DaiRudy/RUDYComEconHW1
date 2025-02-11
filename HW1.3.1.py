import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from numba.experimental import jitclass
from numba import float64

# Define the WealthDynamics class
wealth_dynamics_data = [
    ('w_hat', float64),
    ('s_0', float64),
    ('c_y', float64),
    ('μ_y', float64),
    ('σ_y', float64),
    ('c_r', float64),
    ('μ_r', float64),
    ('σ_r', float64),
    ('a', float64),
    ('b', float64),
    ('σ_z', float64),
    ('z_mean', float64),
    ('z_var', float64),
    ('y_mean', float64),
    ('R_mean', float64)
]

@jitclass(wealth_dynamics_data)
class WealthDynamics:
    def __init__(self, w_hat=1.0, s_0=0.75, c_y=1.0, μ_y=1.0, σ_y=0.2,
                 c_r=0.05, μ_r=0.1, σ_r=0.5, a=0.5, b=0.0, σ_z=0.1):

        self.w_hat, self.s_0 = w_hat, s_0
        self.c_y, self.μ_y, self.σ_y = c_y, μ_y, σ_y
        self.c_r, self.μ_r, self.σ_r = c_r, μ_r, σ_r
        self.a, self.b, self.σ_z = a, b, σ_z

        # Compute stationary moments
        self.z_mean = b / (1 - a)
        self.z_var = σ_z**2 / (1 - a**2)
        exp_z_mean = np.exp(self.z_mean + self.z_var / 2)
        self.R_mean = c_r * exp_z_mean + np.exp(μ_r)
        self.y_mean = c_y * exp_z_mean + np.exp(μ_y)

        # Ensure stability condition (prevent wealth from diverging)
        α = self.R_mean * self.s_0
        if α >= 1:
            raise ValueError("Stability condition failed.")

    def update_states(self, w, z):
        """
        Update wealth dynamics for one period given current wealth (w) and AR(1) state (z).
        """
        zp = self.a * z + self.b + self.σ_z * np.random.randn()
        
        # Compute labor income y_t
        y = self.c_y * np.exp(zp) + np.exp(self.μ_y * np.random.randn() )
        
        # Compute return on assets 1 + r_t
        R = self.c_r * np.exp(zp) + np.exp(self.μ_r * np.random.randn() )
        
        # Apply savings function
        s_w = self.s_0 * w if w >= self.w_hat else 0
        
        # Wealth update equation (without upper bound constraint)
        wp = ( R) * s_w + y
        
        return wp, zp  # No upper bound constraint on wealth

@jit(nopython=True, parallel=True)
def simulate_wealth_distribution(wdy, N, T, w_0):
    """
    Simulates the stationary distribution of wealth for N individuals over T time periods.
    """
    w = np.copy(w_0)  # Copy initial wealth values
    z = wdy.z_mean + np.sqrt(wdy.z_var) * np.random.randn(N)  # Initialize Z

    for t in range(T):
        for i in prange(N):
            w[i], z[i] = wdy.update_states(w[i], z[i])  # Update wealth and state

    return w

# Define simulation parameters
N = 10000  # Number of individuals
T = 1000    # Number of time steps for convergence
w_0 = np.random.uniform(1.0, 10.0, N)  # Initial wealth values randomly distributed

# Initialize the WealthDynamics model
wdy = WealthDynamics()

# Run the wealth simulation without an upper bound constraint
w_stationary = simulate_wealth_distribution(wdy, N, T, w_0)

# Plot the stationary distribution of wealth without constraint
plt.figure(figsize=(8, 6))
plt.hist(w_stationary, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel("Wealth (W)")
plt.ylabel("Density")
plt.title("Stationary Distribution of Wealth (W) without Upper Bound")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()