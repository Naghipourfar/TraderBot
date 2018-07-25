import lstm
import time
import matplotlib.pyplot as plt
import numpy as np
import keras


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


# Main Run Thread
if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 20
    seq_len = 50

    print('> Loading data... ')

    X_train, y_train, X_test, y_test = lstm.load_data_tmp('./data.csv', seq_len, True)

    print('> Data Loaded. Compiling...')

    X_train = np.reshape(X_train, (X_train.shape[0] // seq_len, seq_len, -1))
    y_train = np.reshape(y_train, (y_train.shape[0] // seq_len, seq_len))
    X_test = np.reshape(X_test, (X_test.shape[0] // seq_len, seq_len, -1))
    y_test = np.reshape(y_test, (y_test.shape[0] // seq_len, seq_len))

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # model = lstm.build_model([5, 50, 200, 50])
    # #
    # model.fit(
    #     X_train,
    #     y_train,
    #     batch_size=128,
    #     nb_epoch=epochs,
    #     validation_split=0.15,
    #     verbose=2)

    # model.save('predictor50.h5')

    model = keras.models.load_model("./predictor50.h5")
    #
    predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    # # predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    # # predicted = lstm.predict_point_by_point(model, X_test)
    # # print(len(predicted))
    print('Training duration (s) : ', time.time() - global_start_time)
    plot_results_multiple(predictions, y_test, 50)
