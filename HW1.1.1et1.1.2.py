import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Define parameters
n_states = 11
rho = 0.9  # Autoregressive coefficient
sigma_u = 0.1  # Standard deviation of shock
sigma_y = np.sqrt(1 / 19)  # Standard deviation of y

# Step 2: Discretize state space
y_min = -3 * sigma_y
y_max = 3 * sigma_y
y_grid = np.linspace(y_min, y_max, n_states)

# Step 3: Construct transition matrix using Tauchen method
transition_matrix = np.zeros((n_states, n_states))
for i in range(n_states):
    for j in range(n_states):
        if j == 0:
            transition_matrix[i, j] = norm.cdf((y_grid[j] - rho * y_grid[i] + (y_grid[1] - y_grid[0]) / 2) / sigma_u)
        elif j == n_states - 1:
            transition_matrix[i, j] = 1 - norm.cdf((y_grid[j] - rho * y_grid[i] - (y_grid[1] - y_grid[0]) / 2) / sigma_u)
        else:
            transition_matrix[i, j] = norm.cdf((y_grid[j] - rho * y_grid[i] + (y_grid[1] - y_grid[0]) / 2) / sigma_u) - \
                                     norm.cdf((y_grid[j] - rho * y_grid[i] - (y_grid[1] - y_grid[0]) / 2) / sigma_u)

# Step 4: Compute steady-state distribution from Markov chain
eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
steady_state_distribution = np.real(eigvecs[:, np.isclose(eigvals, 1)]).flatten()
steady_state_distribution = steady_state_distribution / steady_state_distribution.sum()  # Normalize
steady_state_cdf = np.cumsum(steady_state_distribution)  # Compute CDF

# Step 5: Monte Carlo Simulation for y_t process
T = 100000  # Number of steps
burn_in = 1000  # Burn-in period
y_sim = np.zeros(T)
y_sim[0] = 0  # Initialize at 0

for t in range(1, T):
    y_sim[t] = rho * y_sim[t-1] + np.random.normal(0, sigma_u)

y_stationary = np.sort(y_sim[burn_in:])  # Remove burn-in period and sort
simulated_cdf = np.arange(1, len(y_stationary) + 1) / len(y_stationary)  # Compute empirical CDF

# Step 6: Compute theoretical normal CDF
x_vals = np.linspace(min(y_stationary), max(y_stationary), 1000)
theoretical_cdf = norm.cdf(x_vals, loc=0, scale=sigma_y)

# Step 7: Plot all CDFs together
plt.figure(figsize=(8, 6))

# 1. Discrete Markov chain CDF
plt.step(y_grid, steady_state_cdf, where='mid', color='green', linewidth=2, label="Discrete Markov Chain CDF")

# 2. Monte Carlo simulated CDF
plt.plot(y_stationary, simulated_cdf, color='blue', linestyle='--', linewidth=2, label="Simulated CDF")

# 3. Theoretical Normal CDF
plt.plot(x_vals, theoretical_cdf, 'r-', linewidth=2, label="Theoretical N(0, 1/19) CDF")

# Labels and title
plt.xlabel("$y$")
plt.ylabel("Cumulative Probability")
plt.title("Comparison of Steady-State Cumulative Distributions")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
