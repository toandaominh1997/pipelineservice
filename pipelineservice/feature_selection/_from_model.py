import pandas as pd

class selectFromModel(SelectFromModel):
    def transform(self, X):
        data = X.loc[:, self.get_support()]
        cols = data.columns.tolist()
        return pd.DataFrame(data, columns = cols, index = X.index)

