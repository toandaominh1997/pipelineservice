import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion

class featureUninon(FeatureUnion):
    def _hstack(self, Xs):
        cols = [X.columns.tolist() for X in Xs]
        cols = np.hstack(cols)
        return pd.DataFrame(super()._hstack(Xs), columns = cols)
