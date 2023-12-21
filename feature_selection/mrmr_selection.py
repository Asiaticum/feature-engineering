from mrmr import mrmr_classif
from sklearn.base import BaseEstimator, TransformerMixin


class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features):
        self.n_features = n_features

    def fit(self, X, y):
        # MRMRで特徴量を選択
        self.selected_features_ = mrmr_classif(X, y, K=self.n_features)
        print(f"Selected features by MRMR: {self.selected_features_}")
        return self

    def transform(self, X):
        # 選択された特徴量のみを含むデータフレームを返す
        return X[self.selected_features_]
