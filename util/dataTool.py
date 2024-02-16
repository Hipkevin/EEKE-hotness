import pandas as pd
import numpy as np

from . import timer


@timer
def getWordFrequency(csv_path, threshold=15, window_size=3):
    data = pd.read_csv(csv_path)

    data['sum'] = data.iloc[:, 1:].eq(0).sum(axis=1)
    time_series = data[data['sum'] <= threshold]

    data_matrix = time_series.iloc[:, 1: -1].values

    train_set, test_set = [], []

    sample_num, years = data_matrix.shape
    for i in range(window_size - 1, years):
        item = data_matrix[:, i - window_size + 1: i + 1]

        if i == years - 1:
            test_set.append(item)

        else:
            train_set.append(item)

    train_set = np.vstack(train_set)
    test_set = np.vstack(test_set)

    return train_set, test_set
