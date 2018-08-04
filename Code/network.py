import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

"""
    Created by Mohsen Naghipourfar on 8/5/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def plot(x=None, y=None, figpath=None, xlabel="", ylabel="", title=""):
    plt.figure(figsize=(15, 10))
    if x is not None:
        plt.plot(x, y)
    else:
        plt.plot(np.arange(0, y.shape[0]), y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if figpath is not None:
        plt.savefig(figpath)
        plt.close("all")
    else:
        plt.show()


def load_data(data_path="../Data/sp500.csv", results_path="../Results/SP500/", n_seq=50, n_out=1):
    print("Loading Data...")
    data = pd.read_csv(data_path)
    print("Data has been Loaded!")
    print("Data's head")
    print(data.head(2))
    print("Data's Shape\t:\t", data.shape)
    plot(data['DATE'], data['SP500'], figpath=results_path + "data.png", xlabel="Date", ylabel="SP500")
    data = data['SP500']
    data = data.values  # Convert to numpy ND-Array
    data = create_seq_data(data, n_seq, n_out)

    train_data, test_data = train_test_split(data, train_size=0.75)
    train_data, test_data, scaler = scale_data(train_data, test_data)

    x_train, x_test = train_data[:, :-n_out], test_data[:, :-n_out]
    y_train, y_test = train_data[:, -n_out:], test_data[:, -n_out:]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print("x_train has shape\t:\t", x_train.shape)
    print("y_train has shape\t:\t", y_train.shape)

    print("x_test has shape\t:\t", x_test.shape)
    print("y_test has shape\t:\t", y_test.shape)

    return x_train, x_test, y_train, y_test, scaler


def create_seq_data(data, seq_len, n_out=1):
    sequence_length = seq_len + 1
    seq_data = []
    for index in range(len(data) - sequence_length - n_out + 1):
        seq_data.append(data[index: index + sequence_length + n_out])
    seq_data = np.array(seq_data)
    return seq_data


def scale_data(train_data, test_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data, scaler


def train_test_split(data, train_size=0.75):
    n_samples = data.shape[0]
    train_end = int(train_size * n_samples)

    train_data = data[np.arange(0, train_end), :]
    test_data = data[np.arange(train_end, n_samples), :]

    return train_data, test_data


def create_LSTM(n_timestamp, n_features, layers, n_outputs):
    model = Sequential()
    model.add(LSTM(layers[0], input_shape=(n_timestamp, n_features), return_sequences=True))
    model.add(Dropout(0.5))
    for layer in layers[1:-1]:
        model.add(LSTM(layer, return_sequences=True))
    model.add(LSTM(layers[-1], return_sequences=False))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def inverse_scale(x_data, y_data, scaler, n_out=1):
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1]))
    y_data = np.reshape(y_data, (y_data.shape[0], n_out))
    data = np.concatenate((x_data, y_data), axis=1)
    inv_data = scaler.inverse_transform(data)
    return inv_data[:, -n_out:]


def plot_actual_with_predictions(actual, prediction, n_out=1, n_steps=10, figpath=None, xlabel="", ylabel="", title=""):
    plt.figure(figsize=(15, 10))
    plt.plot(actual, color="blue", label="actual")
    if n_out == 1:
        plt.plot(prediction, color="red", label="prediction")
    else:
        start_idx = 0
        for i in range(0, prediction.shape[0], n_steps):
            if i < prediction.shape[0]:
                x = np.arange(start_idx, start_idx + prediction.shape[1])
                plt.plot(x, prediction[i], '-r', color="red")
                start_idx += n_steps
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(figpath)
    plt.close("all")


def main():
    results_path = "../Results/SP500/"
    n_seq = 7
    n_out = 7
    n_steps = 200
    x_train, x_test, y_train, y_test, scaler = load_data(n_seq=n_seq, n_out=n_out)

    model = create_LSTM(x_train.shape[1], x_train.shape[2], [10, 10], n_out)
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=128,
        epochs=20,
        shuffle=False,
        validation_data=(x_test, y_test),
        verbose=2)

    y_test_forecast = model.predict(x_test)

    y_test = inverse_scale(x_test, y_test, scaler, n_out)
    y_test_forecast = inverse_scale(x_test, y_test_forecast, scaler, n_out)

    plot(x=y_test,
         y=y_test_forecast,
         figpath=results_path + "prediction-test.png",
         xlabel="Actual Value",
         ylabel="Predicted Value")

    plot(x=None,
         y=y_test_forecast,
         figpath=results_path + "prediction.png",
         xlabel="Time",
         ylabel="Predicted Value")

    plot(x=None,
         y=y_test,
         figpath=results_path + "actual.png",
         xlabel="Time",
         ylabel="Actual Value")

    plot_actual_with_predictions(actual=y_test,
                                 prediction=y_test_forecast,
                                 figpath=results_path + "actual-with-prediction.png",
                                 n_out=n_out,
                                 n_steps=n_steps,
                                 xlabel="Time",
                                 ylabel="SP500",
                                 )


if __name__ == '__main__':
    main()
