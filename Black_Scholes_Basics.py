import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import re
import math

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

# Plot the implied volatility for a S/K ratio and time to expiration
def black_scholes_implied_volatility_plot(ticker):

    # Get a list of all the expiration dates
    stock = yf.Ticker(ticker)
    expiration = stock.options

    # Get the current stock price
    stock_price = stock.history(period="max")['Close'].iloc[-1]

    data = {}
    
    sks = []
    ttes = []

    for exp in expiration:
        option_chain = stock.option_chain(exp)
        tte = (dt.datetime.strptime(exp, '%Y-%m-%d') - dt.datetime.now()).days / 252
        if tte not in ttes:
            ttes.append(tte)

        # Get the implied volatility for each option
        for option in option_chain.calls.itertuples():
            iv = black_scholes_implied_volatility(stock_price, option.strike, tte, ANNUAL_INTEREST_RATE, option.lastPrice)
            if option.strike not in sks:
                sks.append(round(stock_price / option.strike, 2))
            
            data[(round(stock_price / option.strike, 2), tte)] = iv

    sklist = []
    ttelist = []
    ivlist = []

    for sk in sks:
        for tte in ttes:
            if (sk, tte) not in data.keys():
                data[(sk, tte)] = 0
            sklist.append(np.log(sk))
            ttelist.append(np.log(tte))
            ivlist.append(data[(sk, tte)])


    # Plot the implied volatility for a S/K ratio and time to expiration
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sklist, ttelist, ivlist, c=ivlist, cmap='viridis')
    ax.set_xlabel('S/K Ratio')
    ax.set_ylabel('Time to Expiration')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility for ' + ticker)

    plt.show()