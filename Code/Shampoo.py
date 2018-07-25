import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from math import sqrt
from matplotlib import pyplot
from numpy import array

"""
    Created by Mohsen Naghipourfar on 7/25/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


# date-time parsing function for loading the dataset
def parser(x):
    return pd.datetime.strptime('190' + x, '%Y-%m')


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.DataFrame(diff)


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq, n_features):
    # extract raw values
    raw_values = series.values
    supervised = series_to_supervised(raw_values, n_lag, n_seq)
    supervised_values = supervised.values

    # transform data to be stationary
    #     diff_series = difference(raw_values, 1)
    #     diff_values = diff_series.values
    # diff_values = diff_values.reshape(len(diff_values), 1)

    # rescale values to -1, 1
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(supervised_values)
    print(scaled_values)
    # print(scaled_values.shape)
    # scaled_values = scaled_values.reshape(len(scaled_values), 1)

    # transform into supervised learning problem X, y
    X, y = get_X_y(scaled_values, n_lag, n_seq, n_features, 5)
    y_scaler.fit_transform(y)

    # split into train and test sets
    train, test = scaled_values[0:-n_test], scaled_values[-n_test:]
    return scaler, train, test, y_scaler


def get_X_y(data, n_lag, n_seq, n_features, y_col_idx):
    y_columns = [n_features * (n_lag) + y_col_idx + n_features * i for i in range(n_seq)]
    print(y_columns)
    y = data[:, y_columns]
    X = np.delete(data, y_columns, axis=1)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = get_X_y(train, n_lag, n_seq, n_features, 5)
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=2, shuffle=False, validation_split=0.3)
    # for i in range(nb_epoch):
    #     model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
    #     model.reset_states()
    return model


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, -1)
    # make forecast
    forecast = model.predict(X)
    # convert to array
    return [x for x in forecast[0, :]]


# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    X, y = get_X_y(test, n_lag, n_seq, n_features, 5)
    for i in range(len(test)):
        #         X, y = test[i, 0:n_features*(n_lag)], test[i, n_features*(n_lag):]
        # make forecast
        forecast = forecast_lstm(model, X[i], n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        # forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform([forecast])
        inv_scale = inv_scale[0, :]
        # invert differencing
        #         index = len(series) - n_test + i - 1
        #         last_ob = series.values[index]
        #         inv_diff = inverse_difference(last_ob, inv_scale)

        inverted.append(inv_scale)
    return inverted


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i + 1), rmse))


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test, n_lag, n_seq, training=False):
    pyplot.figure(figsize=(18, 10))
    pyplot.plot(series.values, label="real Data", color='blue')
    for i in range(len(forecasts)):
        if training:
            off_s = i
            off_e = off_s + len(forecasts[i])
        else:
            off_s = len(series) - n_test + i - 1
            off_e = off_s + len(forecasts[i])
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]  # !!!
        #         yaxis = forecasts[i]
        #         print(xaxis, yaxis)
        if i == 0:
            pyplot.plot(xaxis, yaxis, color='red', label='predicted')
        else:
            pyplot.plot(xaxis, yaxis, color='red')
    pyplot.legend()
    pyplot.show()

# load dataset
series = pd.read_csv('./data.csv', header=None)[[i for i in range(1, 7)]]

# configure
n_seq = 2
n_lag = 1
n_test = 10
n_epochs = 20
n_batch = 1
n_neurons = 100
n_features = 6

# prepare data
scaler, train, test, y_scaler = prepare_data(series, n_test, n_lag, n_seq, n_features)

# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# print(pd.DataFrame(forecasts))

# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, y_scaler, n_test + 2)
actual = [row[n_features*(n_lag):] for row in test]
X, y = get_X_y(test, n_lag, n_seq, n_features, 5)
actual = inverse_transform(series, y, y_scaler, n_test + 2)

np.concatenate()
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)

# training forecasts
train_forecasts = make_forecasts(model, n_batch, train, train, n_lag, n_seq)
train_forecasts = inverse_transform(series, train_forecasts, y_scaler, n_test + 2)
print(pd.DataFrame(train_forecasts).shape)

# plot forecasts
plot_forecasts(series[6], train_forecasts, n_test + 2, n_lag, n_seq, training=True)