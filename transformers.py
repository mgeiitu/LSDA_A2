import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class Column_selection(BaseEstimator, TransformerMixin):
  def init(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X[["Direction", "Speed"]]

    return X


class Drop_na(BaseEstimator, TransformerMixin):
  def init(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X.dropna()
    return X


class Scaling(BaseEstimator, TransformerMixin):
  def init(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X['Speed'] = StandardScaler().fit_transform(X['Speed'].values.reshape(-1,1))
    return X


class Radians(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    vals = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 
        'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    rads = {vals[i]: 22.5*i for i in range(len(vals))}

    X = X.replace({"Direction": rads})

    return X


class VectorizeDir(BaseEstimator, TransformerMixin):
  def __init__(self):
    return

  def fit(self, X, y = None):
    return self

  def transform(self, X, y = None):
    wd = X.pop("Direction")
    wv = X["Speed"] 
    wd_rad = wd * np.pi / 180

    X['u'] = wv*np.cos(wd_rad)
    X['v'] = wv*np.sin(wd_rad)

    return X