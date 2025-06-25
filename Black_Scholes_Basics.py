import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import re

# Look into the basics of theoretical option pricing

ANNUAL_INTEREST_RATE = 0.0435
DAILY_INTEREST_RATE = ANNUAL_INTEREST_RATE / 250

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put

def black_scholes_meshplot_S_T(K, r, sigma):
    S = np.linspace(0.5*K, 1.5*K, 100)
    T = np.linspace(0, 2, 100)
    S, T = np.meshgrid(S, T)
    C = black_scholes_call(S, K, T, r, sigma)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(S, T, C, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Expiration')
    ax.set_zlabel('Call Price')
    ax.set_title('Black-Scholes Call Price Surface')
    plt.show()

def black_scholes_meshplot_S_sigma(T, K, r):
    S = np.linspace(0.5*K, 1.5*K, 100)
    sigma = np.linspace(0, 1, 100)
    S, sigma = np.meshgrid(S, sigma)
    C = black_scholes_call(S, K, T, r, sigma)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(S, sigma, C, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Call Price')
    ax.set_title('Black-Scholes Call Price Surface')
    plt.show()

def black_scholes_meshplot_T_sigma(S, K, r):
    T = np.linspace(0, 2, 100)
    sigma = np.linspace(0, 1, 100)
    T, sigma = np.meshgrid(T, sigma)
    C = black_scholes_call(S, K, T, r, sigma)   

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(T, sigma, C, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_xlabel('Time to Expiration')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Call Price')
    ax.set_title('Black-Scholes Call Price Surface')
    plt.show()

def black_scholes_implied_volatility(S, K, T, r, C):
    # Do a binary search for the implied volatility
    sigma_min = 0.01
    sigma_max = 1
    sigma = (sigma_min + sigma_max) / 2
    margin_of_error = 0.0001
    max_iterations = 1000
    iterations = 0

    # Calculate the Black-Scholes call price
    C_BS = black_scholes_call(S, K, T, r, sigma)
    while np.abs(C_BS - C) > margin_of_error and iterations < max_iterations:
        if C_BS < C:
            sigma_min = sigma
        else:
            sigma_max = sigma
        sigma = (sigma_min + sigma_max) / 2
        C_BS = black_scholes_call(S, K, T, r, sigma)
        iterations += 1
    return sigma