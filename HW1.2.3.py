import numpy as np
import matplotlib.pyplot as plt

# Parameters
rho = 0.9  # Autoregressive coefficient
sigma_L = 0.1  # Low uncertainty standard deviation
sigma_H = 0.2  # High uncertainty standard deviation
num_individuals = 1000  # Number of individuals
num_periods = 21  # Number of time periods

# Poisson-distributed m after t=11 (shifted by +3)
np.random.seed(42)  # Set seed for reproducibility
m_t = np.full(num_periods, 3)  # Default m_t = 3 before t=11
m_t[11:] = np.random.poisson(1, num_periods - 11) + 3  # Poisson distribution + 3 after t=11

# Compute steady-state standard deviations
sigma_y_L = sigma_L / np.sqrt(1 - rho**2)
sigma_y_H = sigma_H / np.sqrt(1 - rho**2)

# Initialize individuals' states
y_sim = np.zeros((num_individuals, num_periods))  # All start from y_0 = 0

# Simulating individual paths with dynamically changing state space
for i in range(num_individuals):
    for t in range(1, num_periods):
        if t < 11:
            sigma_t = sigma_L  # Before shock, use low uncertainty
        else:
            sigma_t = sigma_H  # After shock, increase uncertainty

        epsilon = np.random.normal(0, sigma_t)  # Random shock
        y_sim[i, t] = rho * y_sim[i, t-1] + epsilon

        # Adjust state space bounds dynamically based on Poisson-distributed m after t=11
        state_space_t_min = -m_t[t] * sigma_y_H  # Lower bound
        state_space_t_max = m_t[t] * sigma_y_H  # Upper bound

        # Ensure y stays within the dynamically determined state space bounds
        y_sim[i, t] = np.clip(y_sim[i, t], state_space_t_min, state_space_t_max)

# Plot Poisson-distributed m_t over time
plt.figure(figsize=(12, 4))
plt.plot(range(num_periods), m_t, marker='o', linestyle='-', color='b', label="Poisson-distributed $m_t$ + 3 (after $t=11$)")
plt.axvline(x=11, color='r', linestyle='--', label="Uncertainty Shock at $t=11$")
plt.xlabel("Time Periods (t)")
plt.ylabel("State Space Multiplier $m_t$")
plt.title("Dynamically Changing State Space ($m_t$ from Poisson Distribution after $t=11$)")
plt.legend()
plt.grid()
plt.show()

# Plot simulation for 50 individuals
plt.figure(figsize=(12, 6), dpi=100)
for i in range(50):  # Plot only 50 individuals for clarity
    plt.plot(range(num_periods), y_sim[i, :], alpha=0.5)
plt.axvline(x=11, color='r', linestyle='--', label="Uncertainty Shock at $t=11$ (Expanding State Space)")
plt.xlabel("Time Periods (t)")
plt.ylabel("State Variable (y)")
plt.title("Time Path of 1000 Individuals with Poisson-Driven Expanding State Space (After $t=11$)")
plt.legend()
plt.grid()
plt.show()
