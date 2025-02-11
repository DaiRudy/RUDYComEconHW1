import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameter settings
rho = 0.9  # AR(1) coefficients
sigma_u = 0.1  # Error term standard deviation
sigma_y = np.sqrt(1/19)  # Stationary variance of y_t sqrt(Var(y_t))

# Monte Carlo simulation parameters
T = 100000  # Run time step (to ensure steady state)
burn_in = 1000  # Warm-up period to avoid the impact of initial values

# Initialize y_t
y_sim = np.zeros(T)
y_sim[0] = 0  # Start from 0

# Perform Markov chain simulation
for t in range(1, T):
    y_sim[t] = rho * y_sim[t-1] + np.random.normal(0, sigma_u)

# Delete the previous burn_in period data to ensure convergence to steady state
y_stationary = y_sim[burn_in:]

# Calculate the kernel density estimate (KDE) of the steady-state distribution
plt.figure(figsize=(8, 6))
plt.hist(y_stationary, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black', label="Simulated Distribution")
x_vals = np.linspace(min(y_stationary), max(y_stationary), 1000)
plt.plot(x_vals, norm.pdf(x_vals, loc=0, scale=sigma_y), 'r-', label="Theoretical N(0, 1/19)")
plt.xlabel("$y$")
plt.ylabel("Density")
plt.title("Steady-State Distribution of $y_t$")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()