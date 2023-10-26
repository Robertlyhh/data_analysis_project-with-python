import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas_datareader.data as web
# For reading stock data from yahoo
import pandas_datareader.data as web
import yfinance as yf

yf.pdr_override()

# For time stamps
from datetime import datetime
yf.pdr_override()
endtime = datetime.now()
start = datetime(endtime.year-1,endtime.month,endtime.day)

NFLX = web.get_data_yahoo('NFLX', start = start, end = endtime)
from pandas import Series
df = pd.read_csv("real_estate.csv")
