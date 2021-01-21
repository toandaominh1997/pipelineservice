import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, RobustScaler

class polynomialFeatures(PolynomialFeatures):
    def transform(self, X):
        data = super().transform(X)
        cols = self.get_feature_names(input_features = X.columns.tolist())
        return pd.DataFrame(data, columns = cols, index = X.index)


class removeDuplicate(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.unique_indices_ = None
    def fit(self, X, y = None):
        _, self.unique_indices_ = np.unique(X, axis=1, return_index=True)
        return self
    def transform(self, X, y = None):
        return X.iloc[:, self.unique_indices_]

class standardScaler(StandardScaler):
    def transform(self, X):
        data = super().transform(X)
        return pd.DataFrame(data, columns = X.columns, index = X.index)
class robustScaler(RobustScaler):
    def transform(self, X):
        data = super().transform(X)
        return pd.DataFrame(data, columns = X.columns, index = X.index)
