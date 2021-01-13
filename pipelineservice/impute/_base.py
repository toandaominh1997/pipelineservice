import pandas as pd
from sklearn.impute import SimpleImputer


class simpleImputer(SimpleImputer):
    def transform(self, X):
        data = super().transform(X)
        return pd.DataFrame(data, columns = X.columns, index = X.index)
