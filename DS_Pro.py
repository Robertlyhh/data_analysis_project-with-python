import matplotlib
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# For reading stock data from yahoo
import pandas_datareader.data as web
import yfinance as yf

yf.pdr_override()

# For time stamps
from datetime import datetime

endtime = datetime.now()
start = datetime(endtime.year - 1, endtime.month, endtime.day)

my_stock_list = ['NVDA', 'JPM', 'CVX', 'DIS', 'AMZN']
for stock in my_stock_list:
    globals()[stock] = web.get_data_yahoo(stock, start=start, end=endtime)

# my_closing_prices = web.get_data_yahoo(my_stock_list,start,endtime)['Adj Close']
# Risk analysis
closing_prices = web.get_data_yahoo(my_stock_list, start, endtime)['Adj Close']
returns = closing_prices.pct_change().dropna()
rets = returns.dropna()
area = np.pi * 20
# CODE GOES HERE

def get_visualization():
    # THIS FOLLOWING CODE IS FOR VISUAL PURPOSES ONLY. UNCOMMENT TO SEE THEM ON THE SCATTER PLOT.
    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(
            label,
            xy=(x, y), xytext=(50, 50),
            textcoords='offset points', ha='right', va='bottom',
            arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0', color='blue'))

# Set up time horizon
days = 100

# delta
dt = 1 / days

#grab our mu (drift) from the expected return data got for AAPL
mu = rets.mean()['NVDA']

# grab the volatility of the stock from the std() of the average return
sigma = rets.std()['NVDA']

def risk_an_return():
    mean_return = returns.mean()
    returns_series = pd.Series(returns['AMZN'])
    risk = returns_series.std()
    annualized_risk = risk * (252 ** 0.5)
    plt.figure(figsize=(8, 6))
    plt.scatter(risk, returns['AMZN'], label='Data Points', c='b', marker='o')
    plt.title('Risk-Return Profile')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('return')
    plt.axhline(mean_return, color='r', linestyle='--', label=f'Mean Return: {mean_return:.2%}')
    plt.axvline(annualized_risk, color='g', linestyle='--', label=f'Standard Deviation: {annualized_risk:.2%}')
    plt.legend()
    plt.grid(True)
    plt.show()


def monte_carlo(days,start_price):
    price = np.zeros(days)
    price[0] = start_price

    drift = np.zeros(days)
    shock = np.zeros(days)

    for i in range(1,days):
        epsilon = np.random.normal(loc=0, scale=1000, size=1)
        drift[i] = mu*dt
        shock[i] = sigma*epsilon*dt**0.5
        price[i] += price[i-1]*(drift[i]+shock[i])

    return price

def draw_monte_carlo():
    # Get start price from first closing price

    startprice = NVDA["Adj Close"][0]

    #how many simulations we will be running
    simulations = 100

    for run in range(simulations):
        plt.plot(monte_carlo2(days, startprice))

    plt.xlabel('Days')
    plt.ylabel('Opening Price')
    plt.title('Monte Carlo Analysis - NVDA')


def monte_carlo2(days, startprice):
    price = np.zeros(days)
    price[0] = startprice

    for i in range(1, days):
        # Calculate daily returns using a random drift and shock
        drift = mu * (1 / days)  # Mean daily return
        shock = sigma * np.random.normal(0, 1) * np.sqrt(1 / days)  # Daily volatility

        # Calculate the new price
        price[i] = price[i - 1] * (1 + drift + shock)

    return price


def get_array(startprice):
    runs = 10000
    simulations = np.zeros(runs)
    for run in range(runs):
        simulations[run] = monte_carlo2(days, startprice)[days - 1]
    return  simulations


def get_final_risk(simulations,startprice):
    q = np.percentile(simulations, 5)
    sns.histplot(simulations, bins=200, kde=True)

    #some additional information

    # Starting Price
    plt.figtext(0.6, 0.8, s="Start price: $%.2f" % startprice)
    # Mean ending price
    plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

    # Variance of the price (within 95% confidence interval)
    plt.figtext(0.6, 0.6, "VaR(0.95): $%.2f" % (startprice - q))

    # Display 5% quantile
    plt.figtext(0.15, 0.6, "q(0.95): $%.2f" % q)

    # Plot a line at the 5% quantile result
    plt.axvline(x=q, linewidth=2, color='r')

    # Title
    plt.title(u"Final price distribution for Nvidia Stock after %s days" % days, weight='bold')


def get_final_risk2(startprice, end_prices):
    quantile_5th = np.percentile(end_prices, 5)
    quantile_95th = np.percentile(end_prices, 95)

    # Plot a histogram of the end prices with quantiles
    plt.figure(figsize=(10, 6))
    plt.hist(end_prices, bins=50, edgecolor='k', density=True)
    plt.title('Histogram of End Prices (Monte Carlo Simulation)')
    plt.xlabel('End Price')
    plt.ylabel('Frequency')

    # Highlight the quantiles on the histogram
    plt.axvline(quantile_5th, color='r', linestyle='--', label='5th Percentile')
    plt.axvline(quantile_95th, color='g', linestyle='--', label='95th Percentile')
    plt.legend()

    plt.grid(True)
    plt.show()

    # Display the calculated quantiles
    print(f'5th Percentile (Potential Loss): {quantile_5th:.2f}')
    print(f'95th Percentile (Potential Gain): {quantile_95th:.2f}')
