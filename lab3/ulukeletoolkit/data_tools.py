import numpy as np
import pandas as pd
from typing import List, Tuple


def tokenize(data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    data_ = data.copy()
    if not columns:
        columns = data.columns
    for column in columns:
        e = list(enumerate(data[column].unique()))
        m = {ee[1]: ee[0] for ee in e}
        data_[column] = data[column].apply(lambda x: m[x])
    return data_


def pd_to_dataset(data: pd.DataFrame, y_labels: List[str])\
        -> Tuple[np.ndarray, np.ndarray]:
    y = data[y_labels]
    x_labels = [x for x in data.columns if (x not in y_labels)]
    x = data[x_labels]
    return x.to_numpy(copy=True).astype('float'), y.to_numpy(copy=True).astype('float')


def split_dataset(x: np.ndarray, y: np.ndarray, train: float, test: float)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert train >= 0.0
    assert test >= 0.0
    assert (train + test - 1.0) <= 0.0001
    assert len(x) == len(y)

    sz = len(x)
    train_line = int(sz * train)
    x_train = x[:train_line]
    x_test = x[train_line:]

    y_train = y[:train_line]
    y_test = y[train_line:]

    return x_train, x_test, y_train, y_test


def split_dataset_multiple(x: np.ndarray, y: np.ndarray, parts: int)\
        -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    sz = len(x)
    train_step = int(sz / parts)
    train_line = train_step
    ret = []
    while train_line + train_step <= sz:
        x_train = np.concatenate((x[:train_line - train_step], x[train_line:]), axis=0)
        x_test = x[train_line - train_step: train_line]
        y_train = np.concatenate((y[:train_line - train_step], y[train_line:]), axis=0)
        y_test = y[train_line - train_step: train_line]

        ret.append((x_train, x_test, y_train, y_test))
        train_line += train_step
    return ret
