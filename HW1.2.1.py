import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters for AR(1) process
rho = 0.9  # Autoregressive coefficient
sigma_L = 0.1  # Standard deviation in normal state (low uncertainty)
sigma_H = 0.2  # Standard deviation in high uncertainty state
n = 11  # Number of discrete states
m = 3  # Tauchen range (-3σ to +3σ)

# Correcting standard deviation calculation
sigma_y_L = sigma_L / np.sqrt(1 - rho**2)  # Steady-state standard deviation for low uncertainty
sigma_y_H = sigma_H / np.sqrt(1 - rho**2)  # Steady-state standard deviation for high uncertainty

# Define discrete state space for both uncertainty states
y_L = np.linspace(-m * sigma_y_L, m * sigma_y_L, n)
y_H = np.linspace(-m * sigma_y_H, m * sigma_y_H, n)

delta_L = y_L[1] - y_L[0]  # Step size for low uncertainty
delta_H = y_H[1] - y_H[0]  # Step size for high uncertainty

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
df_P_L = pd.DataFrame(P_L, index=np.round(y_L, 2), columns=np.round(y_L, 2))
df_P_H = pd.DataFrame(P_H, index=np.round(y_H, 2), columns=np.round(y_H, 2))

# Check if matrices are different
are_matrices_identical = np.allclose(P_L, P_H)
print("Are the matrices identical after fixing?", are_matrices_identical)

# Plot heatmaps for transition matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(df_P_L, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
axes[0].set_title("Transition Matrix Heatmap (Low Uncertainty)")
axes[0].set_xlabel("Next State")
axes[0].set_ylabel("Current State")

sns.heatmap(df_P_H, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1])
axes[1].set_title("Transition Matrix Heatmap (High Uncertainty)")
axes[1].set_xlabel("Next State")
axes[1].set_ylabel("Current State")


plt.show()

