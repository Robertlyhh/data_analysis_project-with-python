#Let's go ahead and start with some imports
import matplotlib
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# For reading stock data from yahoo
import pandas_datareader.data as web
import yfinance as yf

def plot_data(company, column):
    company[column].plot(legend=True, figsize=(6,4))


def find_moving_average(company,day):
    company["Adj Close"].rolling(day)

def get_rolling(company, day):
    company["Adj Close MA"] = company["Adj Close"].rolling(day).mean()
    company["Adj Close MA"].rolling(20).mean().plot(legend=True, figsize=(6,4))


def get_daily_change_percentage(company):
    # Calculate percent change
    daily_return = company['Close'].pct_change()

    # Plot the daily return
    plt.figure(figsize=(10, 6))  # Set the figure size
    daily_return.plot(title='Daily Return of the Stock', grid=True)
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.show()
