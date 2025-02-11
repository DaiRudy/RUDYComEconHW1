import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Parameter settings
n_states = 11
rho = 0.9  # Autoregression coefficients
sigma_u = 0.1  # Standard deviation of the error term
sigma_y = sigma_u / np.sqrt(1 - rho**2) # Standard deviation of y
y_min = -3 * sigma_y
y_max = 3 * sigma_y

# Discretize state space
y_grid = np.linspace(y_min, y_max, n_states)
delta_y = y_grid[1] - y_grid[0] # State interval (move outside the loop)

# Construct the transfer matrix
transition_matrix = np.zeros((n_states, n_states))

for i in range(n_states):
    for j in range(n_states):
        if j == 0:
            transition_matrix[i, j] = norm.cdf((y_grid[j] - rho * y_grid[i] + delta_y / 2) / sigma_u)
        elif j == n_states - 1:
            transition_matrix[i, j] = 1 - norm.cdf((y_grid[j] - rho * y_grid[i] - delta_y / 2) / sigma_u)
        else:
            transition_matrix[i, j] = norm.cdf((y_grid[j] - rho * y_grid[i] + delta_y / 2) / sigma_u) - \
                                     norm.cdf((y_grid[j] - rho * y_grid[i] - delta_y / 2) / sigma_u)

# Make sure each row sums to 1
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# Calculate steady-state distribution
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
steady_state_distribution = np.real(eigenvectors[:, np.argmax(eigenvalues)])
steady_state_distribution = steady_state_distribution / steady_state_distribution.sum()

# Output results
print("Transition Matrix:")
print(transition_matrix)
print("\nSteady State Distribution:")
print(steady_state_distribution)

# Draw a steady-state distribution graph
plt.figure(figsize=(10, 6))
plt.bar(y_grid, steady_state_distribution, width=delta_y, align='center', alpha=0.7, color='blue')
plt.title('Steady State Distribution')
plt.xlabel('State')
plt.ylabel('Probability')
plt.grid(True)
plt.show()