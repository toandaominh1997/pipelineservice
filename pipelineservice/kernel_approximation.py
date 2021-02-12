import numpy as np
import pandas as pd
from sklearn.kernel_approximation import PolynomialCountSketch, RBFSampler, Nystroem, SkewedChi2Sampler, AdditiveChi2Sampler

class polynomialCountSketch(PolynomialCountSketch):
    feature_name = 'pcsketch'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)


class rBFSampler(RBFSampler):
    feature_name = 'rbf'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)
class nystroem(Nystroem):
    feature_name = 'nystroem'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)

class skewedChi2Sampler(SkewedChi2Sampler):
    feature_name = 'skewedchi2'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)


class additiveChi2Sampler(AdditiveChi2Sampler):
    feature_name = 'additivechi2'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)
