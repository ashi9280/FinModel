import numpy as np
import matplotlib.pyplot as plt

def heston_model(S0, kappa, v0, theta, rho, sigma, r, T, dt, num_paths=1):
    num_steps = int(T / dt)

    S = np.zeros((num_paths, num_steps+1))
    v = np.zeros((num_paths, num_steps+1))

    S[:, 0] = S0
    v[:, 0] = v0

    print(S)
    print(v)

    for t in range(1, num_steps+1):
        Z1 = np.random.normal(0, 1, num_paths)
        Z2 = np.random.normal(0, 1, num_paths)

        dW1 = Z1 * np.sqrt(dt)
        dW2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

        # Update variance

        # Euler Maruyama scheme
        dv = kappa*(theta - v[:,t-1])*dt + sigma*np.sqrt(v[:,t-1])*dW2
        v[:,t] = np.maximum(0, v[:,t-1] + dv)

        # (alternative: Milstein scheme)
        # dv = kappa*(theta - v[:,t-1])*dt + sigma*np.sqrt(v[:,t-1])*dW2 + 0.25*sigma**2/np.sqrt(max(v[:,t-1], 1e-8))*(dW2**2 - dt)
        # v[:,t] = v[:,t-1] + dv

        # Update asset price
        dS = r*S[:,t-1]*dt + np.sqrt(v[:,t-1])*S[:,t-1]*dW1
        S[:,t] = S[:,t-1] + dS

    return S, v

# S0 = 100 # Price of underlying asset
# kappa = 1 # Rate at which vt reverts to theta
# v0 = 0.04 # Initial variance
# theta = 0.04 # Long-term variance
# rho = -0.7 # Correlation between Wiener process of asset price and asset price volatility
# sigma = 0.1 # Volatility of volatility
# r = 0.02 # Risk-free rate
# T = 1 # Time to maturity
# dt = 0.01 # Time step
# num_paths = 10 # Number of paths to simulate

# S, v = heston_model(S0, kappa, v0, theta, rho, sigma, r, T, dt, num_paths)

# Savg = np.mean(S, axis=0)
# vavg = np.mean(v, axis=0)

# for i in range(num_paths):
#     plt.plot(S[i,:], alpha=0.5)
# plt.plot(Savg)
# plt.show()

# for i in range(num_paths):
#     plt.plot(v[i,:], alpha=0.5)
# plt.plot(vavg)
# plt.show()