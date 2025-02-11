import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

# Step 4: Display transition matrix
df_P = pd.DataFrame(P, index=np.round(y_grid, 2), columns=np.round(y_grid, 2))
print("Transition Probability Matrix (Rouwenhorst):")
print(df_P)

# Step 5: Display stationary distribution
df_stationary = pd.DataFrame({"y_grid": y_grid, "Stationary Distribution": stationary_dist})
print("\nStationary Distribution:")
print(df_stationary)

# Step 6: Plot the transition matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(P, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=np.round(y_grid, 2), yticklabels=np.round(y_grid, 2))
plt.title("Transition Matrix Heatmap (Rouwenhorst Approximation)")
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.show()

# Step 7: Plot the stationary distribution
plt.figure(figsize=(8, 6))
plt.bar(y_grid, stationary_dist, width=(y_grid[1] - y_grid[0]), alpha=0.7, color='blue', edgecolor='black')
plt.xlabel("$y_i$ (Discretized states)")
plt.ylabel("Stationary Probability")
plt.title("Stationary Distribution of $y_t$ using Rouwenhorst Approximation")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
