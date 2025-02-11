import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D drawing
from numba import jit, prange
from numba.experimental import jitclass
from numba import float64

# ------------------------------------------------
# 1. Define the WealthDynamics class
# ------------------------------------------------
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
        self.R_mean = self.c_r * exp_z_mean + np.exp(self.μ_r)
        self.y_mean = self.c_y * exp_z_mean + np.exp(self.μ_y)

        # Ensure stability condition (prevent wealth from diverging)
        α = self.R_mean * self.s_0
        if α >= 1:
            raise ValueError("Stability condition failed.")

    def update_states(self, w, z):
        """
        Update the next wealth and status based on the current wealth (w) and status (z)
        """
        zp = self.a * z + self.b + self.σ_z * np.random.randn()
        
        # Calculate labor income y_t
        y = self.c_y * np.exp(zp) + np.exp(self.μ_y * np.random.randn())
        
        # Calculate the asset return rate R (already contains 1, i.e. 1+r_t)
        R = self.c_r * np.exp(zp) + np.exp(self.μ_r * np.random.randn())
        
        # Savings function: Only when wealth is greater than the threshold, save proportionally
        s_w = self.s_0 * w if w >= self.w_hat else 0
        
        # Wealth update formula (no upper bound constraint)
        wp = R * s_w + y
        
        return wp, zp

# ------------------------------------------------
# 2. Define simulation function (single-period update, using parallel acceleration)
# ------------------------------------------------
@jit(nopython=True, parallel=True)
def simulate_wealth_distribution(wdy, N, T, w_0):
    """
    Simulate the wealth distribution of N individuals at T time points
    """
    w = np.copy(w_0)  # Copy the initial wealth array
    z = wdy.z_mean + np.sqrt(wdy.z_var) * np.random.randn(N)  # Initialization state Z

    for t in range(T):
        for i in prange(N):
            w[i], z[i] = wdy.update_states(w[i], z[i])  # Update wealth and status

    return w

# ------------------------------------------------
# 3. Set parameters & run unidimensional distribution simulation
# ------------------------------------------------
N = 10000  # Number of individuals
T = 500    # Number of simulation time points (enough to reach steady state)
w_0 = np.random.uniform(1.0, 10.0, N)  # Randomly generate initial wealth

# Initialize the model
wdy = WealthDynamics()

# Run the simulation (returns the steady-state wealth distribution)
w_stationary = simulate_wealth_distribution(wdy, N, T, w_0)

# ------------------------------------------------
# 4. Generate (w, y) data and plot it in 3D
# ------------------------------------------------
# Since simulate_wealth_distribution only returns w,
# Here we manually replicate a loop synchronized with update_states to save y
w_final = np.copy(w_0)
z_final = wdy.z_mean + np.sqrt(wdy.z_var) * np.random.randn(N)
y_final = np.zeros(N)

for t in range(T):
    for i in range(N):
        # Replicate the update process of update_states
        zp = wdy.a * z_final[i] + wdy.b + wdy.σ_z * np.random.randn()
        # Calculate the new labor income y
        y_final[i] = wdy.c_y * np.exp(zp) + np.exp(wdy.μ_y * np.random.randn())
        # Calculate asset return rate R (consistent with update_states)
        R = wdy.c_r * np.exp(zp) + np.exp(wdy.μ_r * np.random.randn())
        # Savings: Save only when wealth exceeds a threshold
        s_w = wdy.s_0 * w_final[i] if w_final[i] >= wdy.w_hat else 0
        # Update wealth and status (use R directly, without adding 1)
        w_final[i] = R * s_w + y_final[i]
        z_final[i] = zp

# Construct a 2D histogram using (w, y) data and plot it in 3D
bins = 50
H, w_edges, y_edges = np.histogram2d(w_final, y_final, bins=bins)
w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
Wc, Yc = np.meshgrid(w_centers, y_centers)
H_for_plot = H.T  # Transpose to ensure the correct coordinate correspondence

# Drawing 3D Surfaces
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Wc, Yc, H_for_plot, cmap='viridis', edgecolor='none')
ax.set_xlabel('Wealth (W)')
ax.set_ylabel('Labor Income (Y)')
ax.set_zlabel('Frequency')
ax.set_title('3D Histogram: (W, Y) Distribution')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# ------------------------------------------------
# 5. Generate (w, z) data and plot in 3D
# ------------------------------------------------
bins = 50
H_wz, w_edges, z_edges = np.histogram2d(w_final, z_final, bins=bins)
w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
Wc_wz, Zc_wz = np.meshgrid(w_centers, z_centers)
H_wz_for_plot = H_wz.T

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Wc_wz, Zc_wz, H_wz_for_plot, cmap='viridis', edgecolor='none')
ax.set_xlabel('Wealth (W)')
ax.set_ylabel('State Variable (Z)')
ax.set_zlabel('Frequency')
ax.set_title('3D Histogram: (W, Z) Distribution')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
