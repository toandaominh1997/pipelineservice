from feature_engine import EqualFrequencyDiscretiser, EqualWidthDiscretiser, DecisionTreeDiscretiser

class equalFrequencyDiscretiser(EqualFrequencyDiscretiser):
    feature_name = 'equalfrequency'
    def transform(self, X):
        data = super().transform(X)
        data.columns = [f'{self.feature_name}_{col}' for col in data.columns]
        return data

class equalWidthDiscretiser(EqualWidthDiscretiser):
    feature_name = 'equalwidth'
    def transform(self, X):
        data = super().transform(X)
        data.columns = [f'{self.feature_name}_{col}' for col in data.columns]
        return data


class decisionTreeDiscretiser(DecisionTreeDiscretiser):
    feature_name = 'decisiontree'
    def transform(self, X):
        data = super().transform(X)
        data.columns = [f'{self.feature_name}_{col}' for col in data.columns]
        return data

