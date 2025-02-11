import numpy as np
import scipy.stats as stats
import pandas as pd

# Parameters for AR(1) process
rho = 0.9  # Autoregressive coefficient
sigma_L = 0.1  # Standard deviation in normal state (low uncertainty)
sigma_H = 0.2  # Standard deviation in high uncertainty state
n = 11  # Number of discrete states
m = 3  # Tauchen range (-3σ to +3σ)

# Compute steady-state standard deviations
sigma_y_L = sigma_L / np.sqrt(1 - rho**2)
sigma_y_H = sigma_H / np.sqrt(1 - rho**2)

# Define discrete state space for both uncertainty states
y_L = np.linspace(-m * sigma_y_L, m * sigma_y_L, n)
y_H = np.linspace(-m * sigma_y_H, m * sigma_y_H, n)
delta_L = y_L[1] - y_L[0]
delta_H = y_H[1] - y_H[0]

# Construct transition probability matrices for both states
P_L = np.zeros((n, n))
P_H = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if j == 0:
            P_L[i, j] = stats.norm.cdf((y_L[j] + delta_L / 2 - rho * y_L[i]) / sigma_L)
            P_H[i, j] = stats.norm.cdf((y_H[j] + delta_H / 2 - rho * y_H[i]) / sigma_H)
        elif j == n - 1:
            P_L[i, j] = 1 - stats.norm.cdf((y_L[j] - delta_L / 2 - rho * y_L[i]) / sigma_L)
            P_H[i, j] = 1 - stats.norm.cdf((y_H[j] - delta_H / 2 - rho * y_H[i]) / sigma_H)
        else:
            P_L[i, j] = stats.norm.cdf((y_L[j] + delta_L / 2 - rho * y_L[i]) / sigma_L) - \
                        stats.norm.cdf((y_L[j] - delta_L / 2 - rho * y_L[i]) / sigma_L)
            P_H[i, j] = stats.norm.cdf((y_H[j] + delta_H / 2 - rho * y_H[i]) / sigma_H) - \
                        stats.norm.cdf((y_H[j] - delta_H / 2 - rho * y_H[i]) / sigma_H)

# Convert to DataFrame for display
df_P_L = pd.DataFrame(P_L, index=[f"y_{i+1}" for i in range(n)], columns=[f"y_{i+1}" for i in range(n)])
df_P_H = pd.DataFrame(P_H, index=[f"y_{i+1}" for i in range(n)], columns=[f"y_{i+1}" for i in range(n)])

print("Transition Matrix (Low Uncertainty):")
print(df_P_L)
print("\nTransition Matrix (High Uncertainty):")
print(df_P_H)
