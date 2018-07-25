from pandas import ewma
import pandas as pd
import numpy as np


class CryptoAnalysis(object):
    def __init__(self, data):
        self.data = data
        if isinstance(self.data, pd.DataFrame):
            print("Data is pandas Dataframe.")
        elif isinstance(self.data, np.ndarray):
            print("Data is numpy array.")
        elif isinstance(self.data, pd.Series):
            print("Data is pandas Series.")
        else:
            raise ValueError("The input data must be numpy array, pandas DataFrame or Series.")

    def SMA(self, period=3):
        pass

    def EMA(self, periods=3):
        pass

    def BBands(self):
        pass


if __name__ == '__main__':
    c = CryptoAnalysis([1, 2, 3])
