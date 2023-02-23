# Load modules
import pprint
import numpy as np
from tabulate import tabulate
import pandas as pd
import yfinance as yf
import talib as ta
import talib.abstract as ta_h
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from arch import arch_model

# Set up
yf.pdr_override()
pd.set_option('display.max_columns', None)
pp = pprint.PrettyPrinter(indent=4)

# Data and State Info
'''
State Variables:
1. RSI - main indicator (10 days, 30 days, rolling function with time period =N)
    * RSI bounded to (0 - 100)  
    
2. Log(Volume) = log(volume) 
    * traded volume is an indicator of market liquidity
    
3. Time 

4. Momentum = abs(delta_i) = abs(V_i_close - V_i_open) ; i = days

5. Realized Short and long term volatility 
    * can range from 10-180 (window size = 10 for short term, 125 for long term)
    * GARCH(1,1) fit to stock prices
    
6. Moving Average - 50, 100 days normalized 
    * r_i,t = r_i_t / sqrt(h_i_t)

Dependent Variables:
Volume - Traded volume 
Price - Adjusted Close price
Return - r_i = p_i/p_i-1 - 1
    * not really helpful for attempting to predict future values
    * very helpful in analyzing volatility of stock
    * if daily returns have a wide distribution
    * stock is more volatile from one day to next
Log(Return) 
    * log returns are time additive


Independent Variables of stock market:
Explanatory (independent) variables -> have direct relationship
 with the dependent variables and state variable 
 
 Gotta Skip the rows with NaN Values
'''

def get_data(ticker: str, start, end, window=10, verbose=False)->pd.DataFrame:
    stock_data = pdr.get_data_yahoo(ticker, start, end)
    stock_data.index = stock_data.index.date    # Index converted to date
    # Calculate Daily Percentage Returns
    stock_data["Returns"] = 100 * stock_data["Adj Close"].pct_change()  # Daily Return = 1
    # Calculate Daily Log Returns
    # multiplied by 100 to rescale to facilitate optimization
    stock_data['Log Return'] = np.log(stock_data['Adj Close']).diff().mul(100)

    if verbose: # shows before the nan values are dropped
        print(f"\n ******* {ticker} Stock Data for {len(stock_data)} Trading Days, nan only included in"
              f" verbose *******")
        print(tabulate(stock_data[0:window], headers='keys', tablefmt='psql'))
    return stock_data.dropna()  # Drop NaN values


class states:
    def __init__(self, data):
        self.data = data

    # 1. Get RSI Values given then window
    def calc_RSI(self, window=14, verbose=False) -> pd.Series:
        rsi = ta.RSI(self.data['Adj Close'], window)

        if verbose:
            print("RSI INFO:")
            pp.pprint(ta_h.RSI.info)
            print("\n")
            print(f"******* RSI Values for a {window} Day Window and {len(rsi)} Trading Days *******")
            print(tabulate(rsi.to_frame(name="RSI")[0:window+10], headers="keys", tablefmt='psql'))
            print(type(rsi), "\n")
        rsi.name = f"RSI_{window}"
        return rsi

    # 2. Get the log(volume) of the stock
    def log_volume(self, verbose=False):
        log_vol = np.log(self.data['Volume'])

        if verbose:
            print(f"******* Log(Volume) Values for {len(log_vol)} Trading Days *******")
            print(tabulate(log_vol.to_frame(name="log(Volume)")[0:20], headers="keys", tablefmt='psql'))
            print(type(log_vol), "\n")
        log_vol.name = "log(Volume)"
        return log_vol

    # 3. Get time state variable
    def time_state(self, verbose=False):
        time = self.data.index.to_series()

        if verbose:
            print(f"******* Time Values for {len(time)} Trading Days *******")
            print(tabulate(time.to_frame(name="Time")[0:20], headers="keys", tablefmt='psql'))
            print(type(time), "\n")
        time.name = "Time"
        return time

    # 4. Get momentum state variable
    def momentum(self, window, verbose=False)->pd.Series:
        momentum = ta.MOM(self.data['Adj Close'], window)
        if verbose:
            print("Momentum INFO:")
            pp.pprint(ta_h.MOM.info)
            print("\n")
            print(f"******* Momentum Values for a {window} Day Window and {len(momentum)} Trading Days *******")
            print(tabulate(momentum.to_frame(name="Momentum")[0:window+10], headers="keys", tablefmt='psql'))
            print(type(momentum), "\n")
        momentum.name = f"Momentum_{window}"
        return momentum

    # 5. Get realized short and long term volatility
    def base_volatility(self, verbose=False):
        # fitting the model using the arch model package
        basic_gm = arch_model(self.data['Log Return'], mean='Zero', vol='GARCH', p=1, q=1)
        gm_result = basic_gm.fit(update_freq=4)

        if verbose:
            print(f"******* GARCH(1,1) Model Summary for {len(self.data['Log Return'])} Trading Days *******")
            print(gm_result.summary())
            print(gm_result.params)
            # plot fitted results
            gm_result.plot()
            plt.show()
            print("\n Testing for a 5-period ahead forecast")
            ''' reindex = false uses the future behavior
                reindex = true uses the past behavior
                 h.1 in row "2020-05-29": is a 5-step ahead forecast made
                  using data up to and including that date'''
            gm_forecast = gm_result.forecast(horizon=5, reindex=False)
            print(gm_forecast.residual_variance.iloc[-3:])
        return gm_result


        # 6. Get moving average state variable
    def moving_average(self, short_MA, long_MA, verbose=False):
        short_val = self.data['Adj Close'].rolling(window=short_MA).mean()
        long_val = self.data['Adj Close'].rolling(window=long_MA).mean()

        if verbose:
            print(f"******* Short-term ({short_MA}) and Long-term ({long_MA}) Moving Average Values "
                  f"for {len(short_val)} Trading Days *******")
            print(tabulate(pd.concat([short_val.to_frame(name="Short-term MA"), long_val.to_frame(name="Long-term MA")],
                                     axis=1)[0:20], headers="keys", tablefmt='psql'))
            print(type(short_val), "\n",
                  type(long_val), "\n")
        short_val.name = f"Short-term MA_{short_MA}"
        long_val.name = f"Long-term MA_{long_MA}"
        return short_val, long_val






