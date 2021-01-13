import pandas as pd
from sklearn.feature_selection import SelectFromModel

class selectFromModel(SelectFromModel):
    def transform(self, X):
        data = X.loc[:, self.get_support()]
        cols = data.columns.tolist()
        return pd.DataFrame(data, columns = cols, index = X.index)

