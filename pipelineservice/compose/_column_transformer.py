import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

class columnTransformer(ColumnTransformer):
    def _hstack(self, Xs):
        return pd.concat(Xs, axis = 1)

