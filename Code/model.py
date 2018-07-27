import json
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from keras.layers import Dropout, Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import normalize, MinMaxScaler


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


def normalize_data(data):
    return normalize(data, norm='max', axis=1, copy=True)


def create_model(X, layers):
    model = Sequential()
    model.add(LSTM(layers[0], input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(layers[1]))
    model.add(Dropout(0.5))
    model.add(Dense(layers[2], activation='linear', name="output_layer"))
    model.summary()
    model.compile(loss=keras.losses.mse, optimizer="adam")
    return model


# (None, 1, 3) -> (None, 1, 1000) -> (None, 500) -> (None, 3)
def download_data(coin):
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=' + coin + '&tsym=USD&limit=2000')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    return hist


def load_data(filename, sequence_len=100, n_lag=3, n_seq=3, output_cols=None, test_split=0.2):
    raw_data = pd.read_csv(filename, header=None)

    if raw_data.columns.__contains__('Date'):
        raw_data = raw_data.drop(['date'], axis=1)
    else:
        deleted_cols = [i for i in range(raw_data.shape[1] - 1)]
        # for col in raw_data.columns:
        #     if isinstance(raw_data.iloc[0, col], str):  # is either symbol or date
        #         deleted_cols.append(col)
        raw_data = raw_data.drop(deleted_cols, axis=1)

    raw_data_values = series_to_supervised(raw_data.values, n_in=n_lag, n_out=n_seq, dropnan=True)
    # raw_data_values = raw_data.values
    print(pd.DataFrame(raw_data_values).head())

    scaler = MinMaxScaler((-1, 1))

    scaled_values = scaler.fit_transform(raw_data_values)

    # plt.plot(scaled_values[:, 0])
    # plt.show()

    # normalized_values = normalize(raw_data_values, norm='l2', axis=1)

    print(pd.DataFrame(scaled_values).head(5))

    samples = []
    targets = []
    for index in range(0, scaled_values.shape[0] - sequence_len):
        samples.append(scaled_values[index:index + sequence_len])
        targets.append(scaled_values[index + sequence_len, -1])

    samples = np.array(samples)
    print("data has shape\t:\t", samples.shape)

    if output_cols:
        test_size = int(test_split * samples.shape[0])
        x_data = np.array(samples[:, :, :-1])
        y_data = np.array(targets)

        x_train = x_data[:-test_size]
        y_train = y_data[:-test_size]

        x_test = x_data[test_size:]
        y_test = y_data[test_size:]

        print("x_data has shape\t:\t", x_data.shape)
        print("y_data has shape\t:\t", y_data.shape)

        return x_train, y_train, x_test, y_test, raw_data_values, y_data, scaler
    else:
        test_size = int(test_split * samples.shape[0])
        x_data = np.array(scaled_values[:, :n_lag])
        y_data = np.array(scaled_values[:, n_lag:])

        # x_data = x_data.reshape((x_data.shape[0], 1, x_data.shape[1]))
        # y_data = y_data.reshape((y_data.shape[0], 1, y_data.shape[1]))

        x_train = x_data[:-test_size]
        y_train = y_data[:-test_size]

        x_test = x_data[-test_size:]
        y_test = y_data[-test_size:]

        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        print("x_data has shape\t:\t", x_data.shape)
        print("y_data has shape\t:\t", y_data.shape)

        print("x_train has shape\t:\t", x_train.shape)
        print("y_train has shape\t:\t", y_train.shape)

        print("x_test has shape\t:\t", x_test.shape)
        print("y_test has shape\t:\t", y_test.shape)

        return x_train, y_train, x_test, y_test, x_data, y_data, scaler


def plot(data, predictions):
    plt.plot(data, color="blue", label="data")
    plt.plot(predictions, color="red", label="prediction")
    plt.show()


def plot_seq(data, predictions):
    predictions = np.array(predictions)
    plt.plot(data, color="blue", label="data")
    for i in range(predictions.shape[0]):
        x = np.arange(1606 + i, 1606 + i + 3)
        plt.plot(x, predictions[i], color="red", label="prediction")
    plt.show()


def main():
    filename = "../Data/data.csv"
    n_lag = 5
    n_seq = 3
    if not os.path.exists(filename):
        coin = 'BTC'
        data = download_data(coin)
        data.to_csv(filename)
        x_train, y_train, x_test, y_test, x_data, y_data, scaler = load_data(filename, sequence_len=50,
                                                                             output_cols=None)
    else:
        x_train, y_train, x_test, y_test, x_data, y_data, scaler = load_data(filename,
                                                                             sequence_len=50,
                                                                             n_lag=n_lag,
                                                                             n_seq=n_seq,
                                                                             output_cols=None)
    model = create_model(x_train, layers=[1000, 500, 3])
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              epochs=5,
              verbose=2,
              validation_split=0.1,
              shuffle=False)

    predictions = model.predict(x_test)
    print(pd.DataFrame(predictions).head())
    print(pd.DataFrame(predictions).shape)

    plot_seq(x_data[:, n_lag], predictions)


if __name__ == '__main__':
    main()
