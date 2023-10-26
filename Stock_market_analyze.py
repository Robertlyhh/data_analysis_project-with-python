
import matplotlib
import pandas as pd
from pandas import Series,DataFrame
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
start = datetime(endtime.year-1,endtime.month,endtime.day)
stock_list = ['AAPL','TSLA','GOOG','NVDA','AMZN']
def get_closing_prices():
    web.get_data_yahoo(stock_list,start,endtime)['Adj Close']

for stock in stock_list:
    globals()[stock] = web.get_data_yahoo(stock, start = start, end = endtime)

def plot_data(company, column):
    company[column].plot(legend=True, figsize=(6,4))


def find_moving_average(company,day):
    company["Adj Close"].rolling(day)

def get_rolling(company, day):
    company["Adj Close MA"] = company["Adj Close"].rolling(day).mean()
    company["Adj Close MA"].rolling(20).mean().plot(legend=True, figsize=(6,4))

def daily_change_percentage(company, day):
    Series.pct_change(company["Adj Close"], day)


def get_daily_change_percentage(company):
    # Calculate percent change
    daily_return = company['Close'].pct_change()

    # Plot the daily return
    plt.figure(figsize=(10, 6))  # Set the figure size
    daily_return.plot(title='Daily Return of the Stock', grid=True)
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.show()


def get_return(some_price):
    closing_prices = web.get_data_yahoo(stock_list, start, endtime)['Adj Close']
    returns = closing_prices.pct_change().dropna()
    some_price.pct_change().dropna()


def draw_heatmap_1(data):
    x_labels = stock_list
    y_labels = stock_list

    # Create a heatmap
    sns.heatmap(data, cmap="YlGnBu", annot=True, fmt=".1f", xticklabels=x_labels, yticklabels=y_labels)
    # Set labels for the x and y-axis (optional)
    plt.xlabel("X-Axis Labels")
    plt.ylabel("Y-Axis Labels")

    # Show the plot
    plt.show()


def draw_heatmap_2(data):
    closing_prices = web.get_data_yahoo(stock_list, start, endtime)['Adj Close']
    returns = closing_prices.pct_change().dropna()
    correlation_matrix = returns.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
