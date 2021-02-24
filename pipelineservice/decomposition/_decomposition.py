import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, KernelPCA, TruncatedSVD, IncrementalPCA, LatentDirichletAllocation, MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF, SparsePCA, SparseCoder, FactorAnalysis

class factorAnalysis(FactorAnalysis):
    feature_name = 'factoranalysis'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)


class pCA(PCA):
    feature_name = 'pca'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)

class fastICA(FastICA):
    feature_name = 'fastica'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)

class incrementalPCA(IncrementalPCA):
    feature_name = 'incrementalpca '
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)


class kernelPCA(KernelPCA):
    feature_name = 'kernelpca'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)


class latentDirichletAllocation(LatentDirichletAllocation):
    feature_name = 'incrementalpca'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)


class miniBatchDictionaryLearning(MiniBatchDictionaryLearning):
    feature_name = 'minibatchdictionarylearning'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)


class miniBatchSparsePCA(MiniBatchSparsePCA):
    feature_name = 'minibatchsparsepca'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)

class nMF(NMF):
    feature_name = 'nmf'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)

class sparsePCA(SparsePCA):
    feature_name = 'sparsepca'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)

class sparseCoder(SparseCoder):
    feature_name = 'sparsecoder'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)

class truncatedSVD(TruncatedSVD):
    feature_name = 'truncatedsvd'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)
