import numpy as np
import matplotlib.pyplot as plt

# Parameters (redefined to ensure independence)
rho = 0.9  # Autoregressive coefficient
sigma_L = 0.1  # Low uncertainty standard deviation
sigma_H = 0.2  # High uncertainty standard deviation
num_individuals = 1000  # Number of individuals
num_periods = 21  # Number of time periods

# Initialize individuals' states
y_sim = np.zeros((num_individuals, num_periods))  # All start from y_0 = 0

# Simulating individual paths
for i in range(num_individuals):
    for t in range(1, num_periods):
        if t < 11:
            sigma_t = sigma_L  # Before shock, use low uncertainty
        else:
            sigma_t = sigma_H  # After shock, use high uncertainty

        epsilon = np.random.normal(0, sigma_t)  # Random shock
        y_sim[i, t] = rho * y_sim[i, t-1] + epsilon

# Plot simulation for 50 individuals
plt.figure(figsize=(12, 6), dpi=100)
for i in range(50):  # Plot only 50 individuals for clarity
    plt.plot(range(num_periods), y_sim[i, :], alpha=0.5)
plt.axvline(x=11, color='r', linestyle='--', label="Uncertainty Shock at t=11")
plt.xlabel("Time Periods (t)")
plt.ylabel("State Variable (y)")
plt.title("Time Path of 1000 Individuals Over 21 Periods")
plt.legend()
plt.grid()
plt.show()
