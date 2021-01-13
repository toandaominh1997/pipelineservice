import numpy as np
import pandas as pd
import pyod
from imblearn.over_sampling import SMOTE



class Outlier(SMOTE):
    def __init__(self, methods = ['iforest', 'pca'], contamination = 0.1):
        super().__init__()
        self.contamination = contamination
        self.methods = methods
        self.outlier = []
        if "knn" in self.methods:
            knn = pyod.models.knn.KNN(contamination = self.contamination)
            self.outlier.append(knn)
        if 'iforest' in self.methods:
            iforest = pyod.models.iforest.IForest(contamination= self.contamination)
            self.outlier.append(iforest)

        if 'pca' in self.methods:
            pca = pyod.models.pca.PCA(contamination=self.contamination)
            self.outlier.append(pca)

    def _fit_resample(self, X, y = None):
        X = X.copy()
        if y is not None:
            y = y.copy()
        df = pd.DataFrame()
        df['vote_outlier'] = [0]*X.shape[0]
        for out in self.outlier:
            out.fit(X)
            df['vote_outlier'] += out.predict(X)
        print('[Outlier] Remove: {} rows'.format((df['vote_outlier']==len(self.methods)).sum()))
        choose = np.where(df['vote_outlier']!=len(self.methods), True, False)
        if y is not None:
            y = y[choose]

        X = X[[choose]]
        return X, y
