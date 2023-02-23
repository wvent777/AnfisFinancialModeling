import pandas as pd
import numpy as np
import yfinance as yf
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr


#test = yf.download('AAPL', start='2023-01-01', end='2023-01-24')
yf.pdr_override()

test = pdr.get_data_yahoo('AAPL', start='2023-01-01', end='2023-01-24')
test['Deltas'] = test['Adj Close'].sub(test['Open'], axis=0)
test['Deltas abs'] = test['Deltas'].abs()

print(test)
test_np = test.to_numpy()


print(test_np)


# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
x_qual = np.arange(0, 11, 1)


# Generate fuzzy membership functions
qual_lo = fuzz.trimf(x_qual, [0, 0, 5])
qual_md = fuzz.trimf(x_qual, [0, 5, 10])
qual_hi = fuzz.trimf(x_qual, [5, 10, 10])


# Visualize these universes and membership functions
fig, ax0 = plt.subplots(figsize=(8, 9))

ax0.plot(x_qual, qual_lo, 'b', linewidth=1.5, label='Bad')
ax0.plot(x_qual, qual_md, 'g', linewidth=1.5, label='Decent')
ax0.plot(x_qual, qual_hi, 'r', linewidth=1.5, label='Great')
ax0.set_title('Food quality')
ax0.legend()


# Turn off top/right axes
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.get_xaxis().tick_bottom()
ax0.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
