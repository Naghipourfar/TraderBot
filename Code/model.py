import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.layers import Dropout, Dense, LSTM
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
        ax.plot([i], [],)

    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
    plt.show()



def main():
    coin = 'BTC'



if __name__ == '__main__':
    main()

