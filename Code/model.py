import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests
import json
import time

import keras
import keras.backend as K
from keras.layers import Dropout, Dense, LSTM, Input
from keras.models import Model

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import normalize

from datetime import datetime
from datetime import timedelta


def line_plot(line1, line2, label1=None, label2=None, title=''):
    fig, ax = plt.subplots(1, figsize=(25, 15))
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)

    for i, j in zip(line1, line2):
        ax.plot([], [],)

    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
    plt.show()


def normalize_data(data):
    return normalize(data, norm='l2', axis=1, copy=True)

def create_model(input_features, output_neurons=1): # TODO: Has to be checked!
    input_layer = Input(shape=(input_features, ))
    lstm_layer = LSTM(20, input_shape=(input_features, ))(input_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    dense_layer = Dense(output_neurons, activation=keras.activations.linear)

    model = Model(input_layer, dense_layer)
    model.compile(loss=keras.losses.mape, optimizer=keras.optimizers.nadam)

    return model

def download_data():
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym='+coin+'&tsym=USD&limit=2000')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    hist.head()
    return hist


def main():
    coin = 'BTC'
    data = download_data()
    y_data = data['close']
    x_data = data.drop(['close'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=False)

    model = create_model(data.shape[1], output_neurons=1)

    model.fit(x=x_train,
              y=y_train,
              epochs=50,
              batch_size=128,
              validation_data=(x_test, y_test),
              verbose=2)





if __name__ == '__main__':
    main()

