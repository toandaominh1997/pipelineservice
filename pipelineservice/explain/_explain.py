import sklearn
from sklearn.inspection import permutation_importance, plot_partial_dependence
import shap
import seaborn as sns


class Explained(object):
    def __init__(self,
                pipeline,
                X = None,
                y = None,
                ):
        self.pipeline = pipeline
        self.X = X
        self.y = y
    def get_estimator(self):
        try:
            pipeline = self.pipeline[-1]
        except:
            pipeline =  self.pipeline
        sklearn.utils.validation.check_is_fitted(pipeline)
        return pipeline
    def get_xtransform(self):
        try:
            X_transform = self.pipeline[:-1].transform(self.X)
        except:
            X_transform = self.X
        return X_transform


    def get_feature_importance(self):
        feature_importances = None
        for attr in ("feature_importances_", "coef_"):
            try:
                feature_importances = getattr(self.get_estimator(), attr)
            except:
                continue
        data = pd.DataFrame()
        data['feature_name'] = self.get_xtransform().columns.tolist()
        data['feature_importances'] = feature_importances
        return data
    def plot_feature_importance(self, top_k = None):
        data = self.get_feature_importance()
        data = data.sort_values(by = ['feature_importances'], ascending = False)
        if top_k is not None:
            data = data[:top_k]

        height = int(data.shape[0]*0.3)
        aspect = 12/height
        facegrid = sns.catplot(data = data, y = 'feature_name', x = 'feature_importances', kind = 'bar', height = height, aspect=aspect)

        return facegrid.ax

    def get_permutation_importance(self, n_jobs = -1, **kwargs):

        permutation = permutation_importance(estimator = self.get_estimator(),
                                                         X = self.get_xtransform(),
                                                         y = self.y,
                                                         n_jobs = n_jobs,
                                                         **kwargs
                                                        )
        data = pd.DataFrame()
        data['feature_name'] = self.X.columns.tolist()
        data['permutation_importance'] = permutation.importances_mean
        return data

    def plot_permutation_importance(self, top_k = None):
        data = self.get_permutation_importance()
        data = data.sort_values(by = ['permutation_importance'], ascending = False)
        if top_k is not None:
            data = data[:top_k]

        height = int(data.shape[0]*0.3)
        aspect = 12/height
        facegrid = sns.catplot(data = data, y = 'feature_name', x = 'permutation_importance', kind = 'bar', height = height, aspect=aspect)

        return facegrid.ax
    def plot_partial_dependence(self,
                                features,
                                n_jobs = -1,
                                **kwargs):
        fig = plot_partial_dependence(estimator = self.get_estimator(),
                                      X = self.get_xtransform(),
                                      features = features,
                                      n_jobs = n_jobs,
                                      **kwargs
                                     )
        return fig
    def plot_shap_importance(self):
        explainer = shap.TreeExplainer(self.get_estimator())
        shap_values = explainer.shap_values(self.get_xtransform())
        return shap.summary_plot(shap_values, self.get_xtransform(), plot_type = "bar")
