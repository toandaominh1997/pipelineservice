import numpy as np
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis


class neighborhoodComponentsAnalysis(NeighborhoodComponentsAnalysis):
    feature_name = 'nca'
    def transform(self, X):
        data = super().transform(X)
        cols = [f'{self.feature_name}_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns = cols, index = X.index)

