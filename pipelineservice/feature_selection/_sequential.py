import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector

class sequentialFeatureSelector(SequentialFeatureSelector):
    def transform(self, x):
        mask = self.get_support()
        columns = x.loc[:, mask].columns.tolist()
        data = super().transform(x)
        return pd.DataFrame(data, columns = columns, index = x.index)

