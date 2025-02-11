import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import eig

def rouwenhorst(n, rho, sigma):
    """
    Implements Rouwenhorst (1995) method for approximating an AR(1) process.

    Parameters:
    n     : int - Number of states
    rho   : float - Autoregressive coefficient
    sigma : float - Standard deviation of shocks

    Returns:
    y_grid : np.array - Discretized state space
    P      : np.array - Transition probability matrix
    """

    # Compute state space bounds
    y_max = np.sqrt(n - 1) * sigma / np.sqrt(1 - rho**2)
    y_min = -y_max
    y_grid = np.linspace(y_min, y_max, n)

    # Define transition probability parameters
    p = (1 + rho) / 2
    q = p

    # Initialize transition matrix for n = 2
    P = np.array([[p, 1 - p], [1 - q, q]])

    # Recursively build transition matrix
    for i in range(3, n + 1):
        P_old = P
        P = np.zeros((i, i))

        P[:-1, :-1] += p * P_old  # Upper-left
        P[:-1, 1:] += (1 - p) * P_old  # Upper-right
        P[1:, :-1] += (1 - q) * P_old  # Lower-left
        P[1:, 1:] += q * P_old  # Lower-right

        # Normalize non-boundary rows
        P[1:-1] /= 2

    return y_grid, P

# Step 1: Define parameters
n_states = 11  # Number of states
rho = 0.9  # AR(1) coefficient
sigma_u = 0.1  # Standard deviation of shock
sigma_y = np.sqrt(1 / 19)  # Given standard deviation of y

# Step 2: Compute Rouwenhorst approximation
y_grid, P = rouwenhorst(n_states, rho, sigma_u)

# Step 3: Compute stationary distribution
eigvals, eigvecs = eig(P.T)
stationary_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)]).flatten()
stationary_dist /= stationary_dist.sum()  # Normalize
rouwenhorst_cdf = np.cumsum(stationary_dist)  # Compute cumulative distribution function

# Step 4: Monte Carlo Simulation for y_t process
T = 100000  # Number of steps
burn_in = 1000  # Burn-in period
y_sim = np.zeros(T)
y_sim[0] = 0  # Initialize at 0

for t in range(1, T):
    y_sim[t] = rho * y_sim[t-1] + np.random.normal(0, sigma_u)

y_stationary = np.sort(y_sim[burn_in:])  # Remove burn-in period and sort
y_cdf = np.arange(1, len(y_stationary) + 1) / len(y_stationary)  # Compute empirical CDF

# Step 5: Compute theoretical normal CDF
x_vals = np.linspace(min(y_stationary), max(y_stationary), 1000)
theoretical_cdf = norm.cdf(x_vals, loc=0, scale=sigma_y)

# Step 6: Plot all cumulative distribution functions (CDFs) together
plt.figure(figsize=(8, 6))

# 1. Rouwenhorst method CDF
plt.step(y_grid, rouwenhorst_cdf, where='mid', color='green', linewidth=2, label="Rouwenhorst CDF")

# 2. Monte Carlo simulated CDF
plt.plot(y_stationary, y_cdf, color='blue', linestyle='--', linewidth=2, label="Simulated CDF")

# 3. Theoretical Normal CDF
plt.plot(x_vals, theoretical_cdf, 'r-', linewidth=2, label="Theoretical N(0, 1/19) CDF")

# Labels and title
plt.xlabel("$y$")
plt.ylabel("Cumulative Probability")
plt.title("Comparison of Steady-State Cumulative Distributions")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
