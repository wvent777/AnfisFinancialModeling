import keras.models
import numpy as np
import torch
from sanfis import SANFIS, plottingtools
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
import pandas as pd
from tabulate import tabulate
from arch import arch_model
import matplotlib.pyplot as plt
import datetime as dt
import yahoo_finance_data as yfs
from helper_functions import plot_correlogram
from arch.univariate import ARCH, GARCH, EWMAVariance, ConstantMean, Normal

# set environment
pd.set_option('display.max_columns', None)

# Get 5 year SPY data pre-covid
spy_data = yfs.get_data('SPY', dt.datetime(2015, 1, 1), dt.datetime(2020, 1, 1))
print("Original Length of Data: ", len(spy_data))

spy_states = yfs.states(spy_data)
spy_rsi = spy_states.calc_RSI(window=14)
spy_logVol = spy_states.log_volume()
spy_time = spy_states.time_state()
spy_momentum = spy_states.momentum(window=20)
spy_volatility = spy_states.base_volatility()
spy_short_ma, spy_long_ma = spy_states.moving_average(short_MA=20, long_MA=120)

df2 = pd.concat([spy_rsi, spy_logVol,
                 spy_time, spy_momentum, spy_long_ma,
                 spy_short_ma], axis=1)
# combining df's to adjust null values overall
combined_df = pd.concat([spy_data, df2], axis=1)
# Momentum, MA_20, MA_120, RSI_14 are autoregressive variables
combined_df = combined_df.dropna()

print(tabulate(combined_df[0:10], headers='keys', tablefmt='psql'))

# State function values
spy_vol_values = pd.Series(spy_volatility.conditional_volatility, name="\u03C3_(1,1)")
spy_vol_values.index = spy_data.index
state_values = pd.DataFrame(spy_vol_values)
basic_gm = arch_model(combined_df['Log Return'], mean='Zero', vol='GARCH', p=1, q=1)

# EWMA Volatility
# 2/(w+1)
basic_gm.volatility = EWMAVariance(lam=0.95)  # Lambda estimate given the 20 day window
vol_20 = basic_gm.fit(update_freq=5)
S_vol_20 = pd.Series(vol_20.conditional_volatility, name="\u03C3_20")
basic_gm.volatility = EWMAVariance(lam=0.9915)  # Lambda estimate given the 120 day window
vol_120 = basic_gm.fit(update_freq=5)
S_vol_120 = pd.Series(vol_120.conditional_volatility, name="\u03C3_120")

s_df = pd.DataFrame([spy_vol_values, S_vol_20,
                     S_vol_120, spy_short_ma,
                     spy_long_ma, spy_logVol,
                     spy_time, spy_momentum, spy_rsi]).T
s_df = s_df.dropna()
combined_df.drop(columns=['RSI_14', 'log(Volume)',
                          'Time', 'Momentum_20',
                          'Long-term MA_120',
                          'Short-term MA_20'], inplace=True)

print(tabulate(s_df[0:10], headers='keys', tablefmt='psql'))

spy_forecast = spy_volatility.forecast(horizon=20, reindex=False)

# Plot the S&P 500 Daily Volatility
fig1 = plot_correlogram(combined_df['Log Return'], lags=100, title="S&P 500 Log(Return_Adjusted) Correlogram")
fig2 = plot_correlogram(combined_df['Log Return'].sub(combined_df['Log Return'].mean()).pow(2), lags=100,
                        title='S&P 500 Daily Volatility Correlogram')
# Stacked plot of the data
plt.show()

# Split into train and test
# # S, S_test, S_test, X, X_test, X_valid, y, y_train, y_test
# # 1 trading year is #252
assert len(s_df) == len(combined_df)
df_train = combined_df[0:-252]
df_test = combined_df[-252:]
s_train = s_df[0:-252]
s_test = s_df[-252:]

# Plotting the training and testing data
plt.title('S&P 500 Log(Return) Separation of Training and Testing Data', size=20)
plt.plot(df_train['Log Return'], label='Training Data')
plt.plot(df_test['Log Return'], label='Testing Data', color='orange')
plt.legend()
plt.show()

fig, ax = plt.subplots(6, figsize=(10, 5), sharex=True)
ax[0].plot(s_df['log(Volume)'], 'tab:green', label='log(Volume)')
ax[1].plot(s_df['RSI_14'], 'tab:blue', label="RSI_14")
ax[2].plot(s_df['\u03C3_(1,1)'], 'tab:orange', label="\u03C3_(1,1)")
ax[3].plot(s_df['\u03C3_20'], 'tab:purple', label="\u03C3_20")
ax[4].plot(s_df['\u03C3_120'], 'tab:red', label="\u03C3_120")
ax[5].plot(s_df['Momentum_20'], 'tab:gray', label="Momentum_20")
fig.legend(loc='upper center', ncol=5)
plt.xlabel('Date')
plt.show()


def forecast_anfis(model, X, y, epochs=1000, name='Model'):
    # make sure is inverted
    # Convert to Tensor
    state_names = list(X.columns)
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    # Split into train and test
    X_train = X[0:-252]
    X_test = X[-252:]
    y_train = y[0:-252]
    y_test = y[-252:]
    # convert to tensor
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    # Fit the model
    history = model.fit(train_data=[X_train_t, y_train_t],
                        valid_data=[X_test_t, y_test_t],
                        loss_function=torch.nn.MSELoss(reduction='mean'),
                        optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
                        epochs=epochs,
                        use_tensorboard=True,
                        logdir='logs/tb')
    model.plotmfs(names=state_names, save_path=f"results/{name}_mfs.png")
    # Predict
    y_pred = model.predict(X_t)
    # Plot learning curves
    plottingtools.plt_learningcurves(history, save_path=f"results/{name}_learning_curve.png",
                                     title=f"{name} Learning Curve")
    # Plot prediction
    plottingtools.plt_prediction(y_t, y_pred, save_path=f"results/{name}_prediction.png",
                                 title=f"{name} Prediction")
    # RMSE
    _rmse = RMSE(dtype='float32')
    rmse = _rmse(y_t, y_pred)
    rmse = rmse.numpy()
    print('RMSE: ', rmse)
    return y_pred, rmse


def forecast_sanfis(model, S, X, y, epochs=1000, name='Model'):
    # make sure is inverted
    # Convert to Tensor
    state_names = list(S.columns)
    S_t = torch.tensor(S.values, dtype=torch.float32)
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    # Split into train and test
    S_train = S[0:-252]
    S_test = S[-252:]
    X_train = X[0:-252]
    X_test = X[-252:]
    y_train = y[0:-252]
    y_test = y[-252:]
    # convert to tensor
    S_train_t = torch.tensor(S_train.values, dtype=torch.float32)
    S_test_t = torch.tensor(S_test.values, dtype=torch.float32)
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    # Fit the model
    history = model.fit(train_data=[S_train_t, X_train_t, y_train_t],
                        valid_data=[S_test_t, X_test_t, y_test_t],
                        loss_function=torch.nn.MSELoss(reduction='mean'),
                        optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
                        epochs=epochs,
                        use_tensorboard=True,
                        logdir='logs/tb')
    model.plotmfs(names=state_names, save_path=f"results/{name}_mfs.png")
    # Predict
    y_pred = model.predict(X_t)
    # Plot learning curves
    plottingtools.plt_learningcurves(history, save_path=f"results/{name}_learning_curve.png",
                                     title=f"{name} Learning Curve")
    # Plot prediction
    plottingtools.plt_prediction(y_t, y_pred, save_path=f"results/{name}_prediction.png",
                                 title=f"{name} Prediction")
    # RMSE
    _rmse = RMSE(dtype='float32')
    rmse = _rmse(y_t, y_pred)
    rmse = rmse.numpy()
    print('RMSE: ', rmse)
    return y_pred, rmse


# Setting up membership functions
rsi_mf = {'function': 'bell',
          'n_memb': 3,
          'params': {'c': {'value': [0.0, 50.0, 100.0],
                           'trainable': True},
                     'a': {'value': [1000.0, 300.0, 1000.0],
                           'trainable': False},
                     'b': {'value': [10.0, 2.0, 10.0],
                           'trainable': False}}}

sigmoid_membfunc = {'function': 'sigmoid',
                    'n_memb': 2,
                    'params': {'c': {'value': [0.0, 0.0],
                                     'trainable': True},
                               'gamma': {'value': [-2.5, 2.5],
                                         'trainable': True}}}

# # Vanilla ANFIS
# # Model 1  - RSI_14, Momentum_20
model_1_mf = [rsi_mf, sigmoid_membfunc]
model_1 = SANFIS(membfuncs=model_1_mf, n_input=2)
# loss_functions = torch.nn.MSELoss(reduction='mean')
# optimizer = torch.optim.Adam(model_1.parameters(), lr=0.005)

model_1.plotmfs(bounds=[[0, 100],  # plot bounds for rsi_mf
                        [-2.0, 2.0]],  # plot bounds for sigmoid_membfunc
                save_path='results/model_1_mf_before.png')

X_1 = pd.DataFrame([s_df['RSI_14'], s_df['Momentum_20']]).T
y_1 = pd.DataFrame(combined_df['Log Return'])
y_pred_1, rmse_1 = forecast_anfis(model_1, X_1, y_1, epochs=1000, name="Model_1")

# Model 2 RSI_14, log(Volume)
model_2_mf = [rsi_mf, sigmoid_membfunc]
model_2 = SANFIS(membfuncs=model_2_mf, n_input=2)
X_2 = pd.DataFrame([s_df['RSI_14'], s_df['log(Volume)']]).T
y_2 = pd.DataFrame(combined_df['Log Return'])
y_pred_1, rmse_2 = forecast_anfis(model_2, X_2, y_2, epochs=1000, name="Model_2")

# Model 3 - log(Volume), Momentum_20
model_3_mf = [sigmoid_membfunc, sigmoid_membfunc]
model_3 = SANFIS(membfuncs=model_3_mf, n_input=2, scale='Std')
X_3 = pd.DataFrame([s_df['log(Volume)'], s_df['Momentum_20']]).T
y_3 = pd.DataFrame(combined_df['Log Return'])
y_pred_3, rmse_3 = forecast_anfis(model_3, X_3, y_3, epochs=1000, name="Model_3")

# Model 4 GARCH(1,1), GARCH(20)
model_4_mf = [sigmoid_membfunc, sigmoid_membfunc]
model_4 = SANFIS(membfuncs=model_4_mf, n_input=2, scale='Std')
S_4 = pd.DataFrame([s_df['\u03C3_(1,1)'], s_df['\u03C3_20']]).T
X_4 = pd.DataFrame([s_df['log(Volume)'], s_df['Momentum_20']]).T
y_4 = pd.DataFrame(combined_df['Log Return'])
y_pred_4, rmse_4 = forecast_sanfis(model_4, S_4, X_4, y_4, epochs=1000, name="Model_4")

# Model 5 - sigma_20, sigma_120
model_5_mf = [sigmoid_membfunc, sigmoid_membfunc]
model_5 = SANFIS(membfuncs=model_5_mf, n_input=2, scale='Std')
S_5 = pd.DataFrame([s_df['\u03C3_20'], s_df['\u03C3_120']]).T
X_5 = pd.DataFrame([s_df['log(Volume)'], s_df['Momentum_20']]).T
y_5 = pd.DataFrame(combined_df['Log Return'])
y_pred_5, rmse_5 = forecast_sanfis(model_5, S_5, X_5, y_5, epochs=1000, name="Model_5")



