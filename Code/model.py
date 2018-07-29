import json
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from keras.layers import Dropout, Dense, LSTM, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import normalize, MinMaxScaler

"""
    Created by Mohsen Naghipourfar on 7/25/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


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

    return agg.as_matrix()


def normalize_data(data):
    return normalize(data, norm='max', axis=1, copy=True)


def create_model(X, layers):
    model = Sequential()
    for i in range(len(layers) - 1):
        if i == 0:
            model.add(LSTM(layers[0], input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        elif i == len(layers) - 2:
            model.add(LSTM(layers[i], return_sequences=True))
        else:
            model.add(LSTM(layers[i], return_sequences=True))
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(layers[-1], activation='linear')))
    model.compile(loss=keras.losses.mae, optimizer="adam")
    model.summary()
    return model


# (None, 1, 3) -> (None, 1, 1000) -> (None, 500) -> (None, 3)
def download_data(coin):
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=' + coin + '&tsym=USD&limit=2000')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    return hist


def load_data(filename, sequence_len=100, n_lag=3, n_seq=3, target_idx=5, output_cols=None, test_split=0.1):
    raw_data = pd.read_csv(filename, header=None)

    if raw_data.columns.__contains__('Date'):
        raw_data = raw_data.drop(['date'], axis=1)
    else:
        deleted_cols = [i for i in range(raw_data.shape[1]) if i != target_idx]
        # for col in raw_data.columns:
        #     if isinstance(raw_data.iloc[0, col], str):  # is either symbol or date
        #         deleted_cols.append(col)
        raw_data = raw_data.drop(deleted_cols, axis=1)

    raw_data_values = series_to_supervised(raw_data.values, n_in=n_lag, n_out=n_seq, dropnan=True)

    print(pd.DataFrame(raw_data_values).head())

    scalar = MinMaxScaler((-1, 1))

    test_size = int(test_split * raw_data_values.shape[0])

    train_data = np.array(raw_data_values[:-test_size, :])
    test_data = np.array(raw_data_values[-test_size:, :])

    train_data = scalar.fit_transform(train_data)
    test_data = scalar.transform(test_data)

    plt.figure(figsize=(20, 10))
    x = np.arange(0, raw_data_values.shape[0])
    plt.plot(x[:train_data.shape[0]], train_data[:, 0], label="Training Data")
    plt.plot(x[train_data.shape[0]:], test_data[:, 0], label="Test Data")
    plt.legend(loc="best")
    plt.show()

    data = np.concatenate([train_data, test_data], axis=0)
    x_data = np.array(data[:, :n_lag])
    y_data = np.array(data[:, n_lag:])

    x_train = train_data[:, :n_lag]
    y_train = train_data[:, n_lag:]

    x_test = test_data[:, :n_lag]
    y_test = test_data[:, n_lag:]

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    y_test = y_test.reshape((y_test.shape[0], 1, y_test.shape[1]))


    print("x_data has shape\t:\t", x_data.shape)
    print("y_data has shape\t:\t", y_data.shape)

    print("x_train has shape\t:\t", x_train.shape)
    print("y_train has shape\t:\t", y_train.shape)

    print("x_test has shape\t:\t", x_test.shape)
    print("y_test has shape\t:\t", y_test.shape)

    return x_train, y_train, x_test, y_test, x_data, y_data, scalar


def plot(data, predictions):
    plt.plot(data, color="blue", label="data")
    plt.plot(predictions, color="red", label="prediction")
    plt.show()


def plot_seq(data, predictions, n_seq, start_idx=1606, step=1):
    predictions = np.array(predictions)
    plt.figure(figsize=(20, 10))
    plt.plot(data, color="blue", label="data")
    if n_seq == 1:
        x = np.arange(start_idx, start_idx + predictions.shape[0])
        plt.plot(x, predictions, color='red', label="prediction")
    for i in range(predictions.shape[0]):
        x = np.arange(start_idx + i, start_idx + i + n_seq)
        if n_seq != 1:
            plt.plot(x, predictions[i], color="red", label="prediction")
        else:
            plt.plot(x, predictions[i], 'ro', color="red", label="prediction")
        if step != 1:
            start_idx += step
    # plt.legend(handles=[data_plot, prediction_line], labels=['data', 'prediction'], loc="best")
    plt.ylabel("Scaled Close Price")
    plt.xlabel("Time")
    plt.show()


def main():
    filename = "../Data/data.csv"
    n_lag = 5  # past data to be used
    n_seq = 1  # number of future days to predict
    model_path = "./predictor%d_%d.h5" % (n_seq, n_lag)
    target_idx = 5
    epochs = 5
    batch_size = 64

    tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph/', histogram_freq=0,
                                              write_graph=True, write_images=True)
    if not os.path.exists(filename):
        coin = 'BTC'
        data = download_data(coin)
        data.to_csv(filename)
        x_train, y_train, x_test, y_test, x_data, y_data, scalar = load_data(filename, sequence_len=50,
                                                                             n_lag=n_lag,
                                                                             n_seq=n_seq,
                                                                             target_idx=target_idx,
                                                                             output_cols=None)
    else:
        x_train, y_train, x_test, y_test, x_data, y_data, scalar = load_data(filename,
                                                                             sequence_len=50,
                                                                             n_lag=n_lag,
                                                                             n_seq=n_seq,
                                                                             test_split=0.1,
                                                                             target_idx=target_idx,
                                                                             output_cols=None)

    if not os.path.exists(model_path):
        model = create_model(x_train, layers=[1024, 512, 256, 128, n_seq])
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  # validation_split=0.2,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard],
                  shuffle=True)
        model.save(model_path)
        print("The network has been saved!")
    else:
        print("The network exists. Trying to load ...")
        model = keras.models.load_model(model_path)
        print("The network has been loaded!")

    # candidate_data = x_data[[i for i in range(0, x_data.shape[0], 50)], :]

    # candidate_data = candidate_data.reshape((candidate_data.shape[0], 1, candidate_data.shape[1]))

    candidate_data = x_test

    predictions = model.predict(candidate_data)

    # print(pd.DataFrame(predictions).head())

    plot_seq(y_data[:, 0], predictions.reshape((predictions.shape[0], 1)), n_seq, start_idx=x_train.shape[0], step=1)


if __name__ == '__main__':
    main()
