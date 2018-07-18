import numpy as np
import pandas as pd
from pandas_datareader import data

import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import History, CSVLogger

"""
    Created by Mohsen Naghipourfar on 7/23/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

tickers = ['AAPL', 'MSFT', '^GSPC']  # Apple, Microsoft and S&P500 index

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2016-12-31'

panel_data = data.DataReader('INPX', 'google', start_date, end_date)
''' returns a panel object (3D Object)
    1st dim: various fields of finance -> open, close, high, low, ...
    2nd dim: date
    3rd dim: instrument identifiers 
'''

# df_data = panel_data.to_frame()
all_weekdays = pd.date_range(start_date, end_date, freq='B')

close = panel_data['close']
close = close.reindex(all_weekdays)
close = close.fillna(method='ffill')

short_rolling = close.rolling(window=20).mean()


