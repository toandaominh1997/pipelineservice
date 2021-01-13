import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, features_discretize = None):
        super().__init__()
        self.features_to_discretize = features_discretize
        self.disc = None
    def fit(self, X, y = None):
        X = X.copy()
        if self.features_to_discretize is None:
            self.features_to_discretize = X.columns.tolist()
        if len(self.features_to_discretize) > 0:
            self.binns = []
            for col in self.features_to_discretize:
                hist, _ = np.histogram(X[col], bins = 'sturges')
                self.binns.append(len(hist))

            self.len_columns = len(self.features_to_discretize)
            self.disc = KBinsDiscretizer(n_bins = self.binns,encode = 'ordinal',
                strategy = 'kmeans')
            self.disc.fit(X[self.features_to_discretize].values.reshape(-1, self.len_columns))
        return self
    def transform(self, X, y = None):
        X = X.copy()
        if self.disc is not None:
            X[self.features_to_discretize] = self.disc.transform(X[self.features_to_discretize].values.reshape(-1, self.len_columns))
            data = X[self.features_to_discretize]
            data.columns = [f'bin_{col}' for col in data.columns]
            return data
        return X
