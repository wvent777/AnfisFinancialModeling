import yahoo_finance_data as yfs
import datetime as dt

# Import data
aapl_data = yfs.get_data("AAPL", dt.datetime(2020, 1, 1), dt.datetime(2020, 6, 1), verbose=True)

aapl_states = yfs.states(aapl_data)

# 1. Test RSI
rsi_test = aapl_states.calc_RSI(10, verbose=True)

# 2. Test log volume
log_vol_test = aapl_states.log_volume(verbose=True)

# 3. Test Time
time_test = aapl_states.time_state(verbose=True)

# 4. Test momentum
momentum_test = aapl_states.momentum(window=14, verbose=True)

# 5. Test volatility
volatility_test = aapl_states.base_volatility(verbose=True)

# 6. test moving average
short, long = aapl_states.moving_average(short_MA=20, long_MA=100, verbose=True)
