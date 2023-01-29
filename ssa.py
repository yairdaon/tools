#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:08:37 2018

@author: Vivien Sainte Fare Garnot
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from optht import optht


def embedded(x, M):
    """Computes the M embedding of a time series

    Args:
        x (numpy array): data time series
        M (int): dimension of the desired embedding (window length)

    Returns:
        the embedded trajectory matrix
    """
    N2 = x.shape[0] - M + 1
    X = np.zeros((N2, M))
    for i in range(N2):
        X[i, :] = x[i:i + M]
    return np.matrix(X)


class GSSA:
    """Generic instance of a SSA analysis
    Args:
        data: array like, input time series, must be one dimensional

    Attributes:
        data (array): input time series
        index (index): index of the time series
        M (int): Window length
        N2 (int): Reduced length
        X (numpy matrix): Trajectory matrix
    """

    def __init__(self, M=None):
        self.M = M

    def fit(self, X):
        """Completes the Analysis on a SSA object

        Args:
            M (int): window length

        """
        X = np.array(X)
        X = np.squeeze(X)
        assert len(X.shape) == 1, "Data must be 1D"
        self.N = X.size
        self.N2 = self.N - self.M + 1

        if (self.N2 < self.M):
            raise ValueError('Window length is too big')
        else:
            self.X = embedded(X, self.M)

        self.U, self.s, self.Vt = np.linalg.svd(self.X)
        self.cutoff = optht(self.M / self.N2, sv=self.s)
        return self

    def predict(self):
        S = np.zeros(self.X.shape)
        np.fill_diagonal(S, self.s)

        S[self.cutoff:] = 0
        H = self.U @ S @ self.Vt
        H = np.flipud(H)

        rec = np.empty(self.N)
        for k in range(self.N):
            rec[k] = np.diagonal(H, offset=-(self.N2 - 1 - k)).mean()
        return rec

    @staticmethod
    def reconstruct(data, M):
        if np.unique(data).size == 1:
            return data
        ssa = GSSA(M)
        if type(data) == pd.Series:
            return ssa.fit(data).predict()
        elif type(data) == pd.DataFrame:
            res = pd.DataFrame(index=data.index, columns=data.columns)
            for col in data.columns:
                res[col] = ssa.fit(data[col].values).predict()
            return res
        else:
            raise NotImplemented("Data must be pd.Series or pd.DataFrame")


def main():
    # np.random.seed(11234)
    years = 50
    T = np.linspace(0, years, num=365 * years)
    signal = np.sin(2 * np.pi * 1 * T)
    signal += np.sin(2 * np.pi * 4 * T)
    signal += np.cos(2 * np.pi * 8.3 * T)
    # signal += np.sin(2 * np.pi * 9 * T)
    # signal += np.sin(2 * np.pi * 4.33 * T)
    # signal += np.cos(2 * np.pi * 12 * T)

    noise = np.random.randn(T.size) / 4
    series = pd.Series(index=T, data=signal + noise)

    # series = pd.read_csv("/home/yair/projects/native_EDM/data/israel/rami/ili.csv")['weekly']
    # series = np.log(series)
    # series = pd.read_csv("/home/yair/projects/native_EDM/data/israel/israel.csv").rename({"level_0": 'time'}, axis=1).AH

    n_cycles = 3
    series = series.iloc[::7]
    M = n_cycles * 365 // 7

    auto = GSSA.reconstruct(data=series, M=M)
    res = {'data': series,
           'auto': auto}
    res = pd.DataFrame(index=series.index, data=res)

    res.plot()
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except:
        import pdb, traceback, sys

        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
