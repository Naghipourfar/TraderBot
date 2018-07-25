import json
import os

import keras
import numpy as np
import pandas as pd
import requests
from keras.layers import Dropout, Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import normalize


def normalize_data(data):
    return normalize(data, norm='max', axis=1, copy=True)


def create_model(x_data, layers):
    model = Sequential()
    model.add(LSTM(layers[0], input_shape=(x_data.shape[1], x_data.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(layers[1], return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(layers[-1], activation='linear'))
    model.compile(loss=keras.losses.mse, optimizer="rmsprop")
    return model


def download_data(coin):
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=' + coin + '&tsym=USD&limit=2000')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    return hist


def load_data(filename, sequence_len=100, output_cols=None, test_split=0.2):
    raw_data = pd.read_csv(filename)

    if raw_data.columns.__contains__('Date'):
        raw_data.drop(['date'], axis=1)
    else:
        deleted_cols = []
        # for col in raw_data.columns:
        #     if isinstance(raw_data.iloc[0, col], str):  # is either symbol or date
        #         deleted_cols.append(col)
        raw_data.drop(deleted_cols, axis=1)

    raw_data_values = raw_data.values

    samples = []
    for index in range(0, raw_data.shape[0] - sequence_len):
        samples.append(raw_data_values[index:index + sequence_len])

    samples = np.array(samples)
    print("data has shape\t:\t", samples.shape)

    if output_cols:
        test_size = int(test_split * samples.shape[0])
        x_data = np.array(samples[:, :, :-1])
        y_data = np.array(samples[:, :, -1])

        x_train = x_data[:-test_size]
        y_train = y_data[:-test_size]

        x_test = x_data[test_size:]
        y_test = y_data[test_size:]

        return x_train, y_train, x_test, y_test
    else:
        return None, None, None, None


def main():
    filename = "./data.csv"
    if not os.path.exists(filename):
        coin = 'BTC'
        data = download_data(coin)
        data.to_csv(filename)
        x_train, y_train, x_test, y_test = load_data(filename, sequence_len=50, output_cols=[5])
    else:
        x_train, y_train, x_test, y_test = load_data(filename, sequence_len=50, output_cols=[5])

    model = create_model(x_train, layers=[50, 50, 1])
    model.fit(x=x_train,
              y=y_train,
              batch_size=1,
              epochs=5,
              verbose=2,
              validation_split=0.1,
              shuffle=False)


if __name__ == '__main__':
    main()
