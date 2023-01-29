import warnings

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.feature_selection import mutual_info_regression as mutual_info
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.stattools import acf

from ssa import GSSA


def ssa(df,
        M,
        log_var=None):
    res = {}
    for col in df.columns:
        if 'year' in col:
            continue
        log_transform = (col == log_var)
        x = np.log(df[col]) if log_transform else df[col]
        x = GSSA.reconstruct(data=x, M=M)  # SSA with Gavish's threshold
        res[col] = np.exp(x) if log_transform else x
    return pd.DataFrame(res, index=df.index)


##############################################
## Lagging function ##########################
##############################################
def lag(df,
        lags):
    """Lags df according to lags. Lags are denoted with _underscore_, e.g. X_4.
    Observations, OTOH come with a sign: X-4 or Y+3.
    """
    if type(df) == pd.Series:
        df = pd.DataFrame(index=df.index, columns=[df.name], data=df.values)
    ## If lags is int, we lag all variables 0,...,lags-1. Here we
    ## create the lagging dictionary.
    if isinstance(lags, int):
        assert lags > 0, 'If lags is an integer it has to be > 0'
        dic = {}
        for col in df.columns:
            dic[col] = range(lags)
        lags = dic

    lagged = pd.DataFrame(index=df.index)
    for var, lag_list in lags.items():
        for lag in lag_list:
            assert lag >= 0
            lagged[f"{var}_{lag}"] = df[var].shift(lag)

    assert lagged.shape[0] > 0
    assert lagged.shape[1] > 0
    return lagged


def remove_nan_rows(X, y):
    k = X.shape[1]
    merged = pd.concat([X, y], axis=1).dropna()
    return merged.iloc[:, :k], merged.iloc[:, k:]


##############################################
## Embedding Parameters ######################
##############################################

def find_embedding_dimension(series: pd.Series,
                             tau: int,
                             max_embedding_dimension: int = 10,
                             first_criterion_tolerance: float = 10,
                             second_criterion_tolerance: float = 1,
                             full: bool = False):
    """Find minimal embedding dimension via the false nearest neighbors method of:
    Determining embedding dimension for phase-space reconstruction using a geometrical
    construction. Kennel and Brown, 1992
    """
    if series.std() == 0:
        return {'E': 1} if full else 1
        
    lags = {series.name: tau * np.arange(max_embedding_dimension)}
    lagged = lag(series, lags).dropna()
    typical_length = series.std()

    mean_false_nearest_neighbors = {}
    for e in range(1, max_embedding_dimension):
        cut_lagged = lagged.values[:, :e]
        distances, indices = NearestNeighbors(n_neighbors=2, p=2).fit(cut_lagged).kneighbors(cut_lagged)
        distances, indices = distances[:, 1], indices[:, 1]
        discrepancy = np.abs(lagged.values[indices, e] - lagged.values[:, e])
        first_criterion = discrepancy / distances > first_criterion_tolerance
        second_criterion = np.sqrt(distances ** 2 + discrepancy ** 2) / typical_length > second_criterion_tolerance
        false_nearest_neighbors = first_criterion | second_criterion
        mean_false_nearest_neighbors[e] = np.mean(false_nearest_neighbors)

    mean_false_nearest_neighbors = pd.Series(mean_false_nearest_neighbors)
    ind = mean_false_nearest_neighbors < 0.05
    if ind.any():
        optimal_embedding_dimension = mean_false_nearest_neighbors.loc[ind].index.min()
    else:
        optimal_embedding_dimension = mean_false_nearest_neighbors.index.max()
        warnings.warn(f"Unable to calculate embedding dimension for {series.name}!")
    dic = {'FNN': mean_false_nearest_neighbors, 'E': optimal_embedding_dimension}
    return dic if full else optimal_embedding_dimension


def optimal_tau(series: pd.Series,
                method: str = 'mutual_information',
                full: bool = False,
                span: int = 100):
    """ Finds best tau, either as first location where auto correlation drops below 1/e OR
    as the first minimum of the mutual information function between time series and its lags

    Args:
        max_embedding_dimension:
        full:
        method: for 'acor' we employ a heuristic from slides "State Space Reconstruction" by Ng
        Sook Kien. For 'info' we find the first minimum of the mutual information
        between time series and delayed time series, for each delay tp. See Independent coordinates
         for strange attractors from mutual information, by Fraser and Swinney, 1986
        series: pd.Series
        span (int): Where we search for the first minimum of the mutual information.  """

    if method == 'acor':  # Auto correlation
        auto_correlation = acf(series, nlags=span, fft=True)
        decorrelated = np.where(auto_correlation < 1 / np.e)[0]
        optimal = 1 if decorrelated.size == 0 else int(min(decorrelated))
        auto_correlation = pd.Series(data=auto_correlation,
                                     index=range(auto_correlation.shape[0]))
        ret = {'acor': auto_correlation, 'tau': optimal} if full else optimal

    elif method == 'info': 
        lags = {series.name: list(range(span))}
        lags = lag(series, lags)
        merged = pd.concat([lags, series], axis=1).dropna()
        lags, ts = merged.iloc[:, :-1], merged.iloc[:, -1]
        info = mutual_info(lags, ts)
        info = pd.Series(data=info, index=range(span))
        assert not info.isnull().any()
        peaks = signal.find_peaks(-info.values)[0]
        if len(peaks) == 0:
            optimal = None
        else:
            optimal = info.index[peaks[0]]
        ret = {'info': info, 'tau': optimal} if full else optimal
    else:
        raise ValueError(f"method {method} not implemented")
    return ret


########################################
# Combine all functions above ##########
########################################
def transform_series(series: pd.Series,
                     tau_method: str):
    if series.std() == 0:
        tau = 1
        embedding_dimension = 1
    else:    
        tau = optimal_tau(series, method=tau_method)
        if tau is None and 'info' in tau_method:
            warnings.warn(f"Mutual Information calculation of tau failed for {series.name}, switching to autocorrelation")
            tau_method = 'info->acor'
            tau = optimal_tau(series, method=tau_method)
        embedding_dimension = find_embedding_dimension(series, tau=tau)
    if embedding_dimension <= 1 or tau <= 1:
        msg = f"{series.name}: embedding_dimension={embedding_dimension}, tau={tau}."
        msg += "This typically happens for short simulations."
        warnings.warn(msg)
    #embedding_dimension = max(2, embedding_dimension)

    lags = [tau * E for E in range(embedding_dimension)]
    lagged = lag(series, {series.name: lags}).dropna()

    return lagged, dict(embedding_dimension=embedding_dimension, tau=tau, tau_method=tau_method)
