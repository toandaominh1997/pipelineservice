import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

class columnTransformer(ColumnTransformer):
    def _hstack(self, Xs):
        cols = [X.columns.tolist() for X in Xs]
        cols = np.hstack(cols)
        return pd.DataFrame(super()._hstack(Xs), columns = cols)

