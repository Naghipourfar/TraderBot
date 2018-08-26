import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class PastSampler:
    """
    Forms training samples for predicting future values from past value
    """

    def __init__(self, N, K, sliding_window=True):
        """
        Predict K future sample using N previous samples
        """
        self.K = K
        self.N = N
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K  # Number of samples per row (sample + target)
        # indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0] % M == 0:
                I = np.arange(M) + np.arange(0, A.shape[0], M).reshape(-1, 1)

            else:
                I = np.arange(M) + np.arange(0, A.shape[0] - M, M).reshape(-1, 1)

        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]  # Number of features per sample
        return B[:, :ci], B[:, ci:]  # Sample matrix, Target matrix


# data file path
coins = 'BTC', 'LTC', 'ETH'
for coin in coins:
    dfp = '../Data/' + coin + '.csv'

    # Columns of price data to use
    columns = ['Close', 'High', 'Low']
    df = pd.read_csv(dfp)
    time_stamps = df['Timestamp']
    print(time_stamps.shape)
    df = df.loc[:, columns]
    original_df = pd.read_csv(dfp).loc[:, columns]

    file_name = '../Data/' + coin + '.h5'

    scalar = MinMaxScaler((-1, 1))
    # normalization
    df[columns] = scalar.fit_transform(df[columns].values.reshape(-1, len(columns)))

    # Features are input sample dimensions(channels)
    A = np.array(df)[:, None, :]
    original_A = np.array(original_df)[:, None, :]
    time_stamps = np.array(time_stamps)[:, None, None]

    # Make samples of temporal sequences of pricing data (channel)
    NPS, NFS = 360, 84  # Number of past and future samples
    ps = PastSampler(NPS, NFS, sliding_window=False)
    B, Y = ps.transform(A)
    input_times, output_times = ps.transform(time_stamps)
    original_B, original_Y = ps.transform(original_A)

    with h5py.File(file_name, 'w') as f:
        f.create_dataset("inputs", data=B)
        f.create_dataset('outputs', data=Y)
        f.create_dataset("input_times", data=input_times)
        f.create_dataset('output_times', data=output_times)
        f.create_dataset("original_datas", data=np.array(original_df))
        f.create_dataset('original_inputs', data=original_B)
        f.create_dataset('original_outputs', data=original_Y)
