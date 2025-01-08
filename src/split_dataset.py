import pandas as pd
import numpy as np


def split_dataset_train_test_validate(X: pd.DataFrame, y: pd.Series):
    perm = np.random.permutation(len(y))
    X = X.iloc[perm].reset_index(drop=True)
    y = y.iloc[perm].reset_index(drop=True)

    indeks_validate = int(0.7 * len(y))
    indeks_test = int(0.85 * len(y))

    x_train = X.iloc[:indeks_validate]
    x_train.reset_index(drop=True, inplace=True)
    x_train = x_train.to_numpy()

    y_train = y.iloc[:indeks_validate]
    y_train.reset_index(drop=True, inplace=True)
    y_train = y_train.to_numpy()

    x_validate = X.iloc[indeks_validate:indeks_test]
    x_validate.reset_index(drop=True, inplace=True)
    x_validate = x_validate.to_numpy()

    y_validate = y.iloc[indeks_validate:indeks_test]
    y_validate.reset_index(drop=True, inplace=True)
    y_validate = y_validate.to_numpy()

    x_test = X.iloc[indeks_test:]
    x_test.reset_index(drop=True, inplace=True)
    x_test = x_test.to_numpy()

    y_test = y.iloc[indeks_test:]
    y_test.reset_index(drop=True, inplace=True)
    y_test = y_test.to_numpy()

    return x_train, y_train, x_validate, y_validate, x_test, y_test
