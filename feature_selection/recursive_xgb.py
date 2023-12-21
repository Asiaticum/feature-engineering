import copy
import random
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import DataConversionWarning

np.random.seed(99)
random.seed(42)


warnings.filterwarnings("ignore")

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def warn(*args: Any, **kwargs: Any) -> None:
    pass


class RecursiveXGBoost(BaseEstimator, TransformerMixin):
    """
    Feature Selection class compatible with scikit-learn's Pipeline

    Parameters
    ----------
    model_class : str
        Model class (Regression, Binary_Classification, Multi_Classification)
    multi_label : bool
        Multi label problem
    verbose : int
        Verbosity level
    """

    def __init__(
        self,
        model_class="Binary_Classification",
        multi_label=False,
        feature_importance_rate_limit=0.05,
        subset_number=5,
        verbose=0,
    ):
        self.model_class = model_class
        self.multi_label = multi_label
        self.feature_importance_rate_limit = feature_importance_rate_limit
        self.subset_number = subset_number
        self.verbose = verbose
        self.important_features = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.DataFrame or pd.Series
            Target values

        Returns
        -------
        self : object
            Returns self.
        """
        # Implement the feature selection logic here
        self.X_ = X.copy()
        self.y_ = y.copy()

        self.important_features = self._extract_geatures_by_recursive_xgboost()
        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_r : array-like of shape (n_samples, n_selected_features)
            The array of input samples with only the selected features.
        """
        if self.important_features is None:
            raise ValueError("FeatureSelection has not been fitted yet.")

        return X[self.important_features]

    # Rest of the methods (_set_xgb_config, _determine_iteration_limits, etc.) remain the same as in your original class

    def _extract_geatures_by_recursive_xgboost(self):
        train_p = self.X_.copy()
        y_train = self.y_.copy()
        important_features, bst_models = [], []

        self._set_xgb_config()
        iter_limit, top_num = self._determine_iteration_limits(
            train_p, self.subset_number
        )

        try:
            print("    Performing recursive XGBoost feature selection")
            print(f"    Taking at most {top_num} features from each iteration")
            for i in range(0, train_p.shape[1], iter_limit):
                print(f"    Iteration {i}/{i + iter_limit}")
                X_train = self._select_data_subset(train_p, i, iter_limit)
                if X_train.shape[1] < 2:
                    continue

                num_rounds = 20 if X_train.shape[0] >= 100000 else 100
                self._print_booster_rounds(i, num_rounds)

                params = self._get_xgb_params()
                bst = self._train_xgb_model(X_train, y_train, params, num_rounds)
                bst_models.append(bst)

                imp_feats = self._get_feature_importances(bst, X_train)
                print_feats = self._select_important_features(imp_feats, top_num)
                important_features += print_feats

                self._print_selected_features(print_feats)
                important_features = list(OrderedDict.fromkeys(important_features))
            if self.verbose >= 2:
                self._plot_feature_importances(bst_models)
        except Exception as e:
            print("Error during XGBoost training: %s" % e)
            important_features = copy.deepcopy(train_p.columns.tolist())

        if self.verbose:
            print(f"Using {len(important_features)} features after recursive XGBoost")
        print(f"    Selected by XGB: {important_features}")
        return important_features

    def _set_xgb_config(self):
        try:
            xgb.set_config(verbosity=0)
        except Exception as e:
            if self.verbose:
                print("Failed to set XGBoost config: %s" % e)

    def _determine_iteration_limits(self, train_p, subset_number=5):
        if train_p.shape[1] <= subset_number * 2:
            iter_limit = 2
        else:
            iter_limit = int(train_p.shape[1] / subset_number)

        cols_sel = train_p.columns.tolist()
        if len(cols_sel) <= 50:
            top_num = int(max(2, len(cols_sel) / (subset_number * 2)))
        else:
            top_num = int(len(cols_sel) / (subset_number * 2))
        return iter_limit, top_num

    def _select_data_subset(self, train_p, i, iter_limit):
        if train_p.shape[1] - i < iter_limit:
            X_train = train_p.iloc[:, i:]
        else:
            X_train = train_p.iloc[:, i : i + iter_limit]
        return X_train

    def _print_booster_rounds(self, i, num_rounds):
        if i == 0 and self.verbose:
            print("    Number of booster rounds = %s" % num_rounds)

    def _get_xgb_params(self):
        if self.model_class == "Regression":
            params = {
                "objective": "reg:squarederror",
                "silent": True,
                "verbosity": 0,
                "min_child_weight": 0.5,
            }
        elif self.model_class == "Binary_Classification":
            params = {
                "objective": "binary:logistic",
                "num_class": 1,
                "silent": True,
                "verbosity": 0,
                "min_child_weight": 0.5,
            }
        else:
            num_class = (
                self.X[self.target].nunique()
                if isinstance(self.target, str)
                else self.X[self.target].nunique()[0]
            )
            params = {
                "objective": "multi:softmax",
                "silent": True,
                "verbosity": 0,
                "min_child_weight": 0.5,
                "num_class": num_class,
            }
        return params

    def _train_xgb_model(self, X_train, y_train, params, num_rounds):
        dtrain = xgb.DMatrix(
            X_train, y_train, enable_categorical=True, feature_names=X_train.columns
        )
        bst = xgb.train(params, dtrain, num_boost_round=num_rounds)
        return bst

    def _get_feature_importances(self, bst, X_train):
        imp_feats = bst.get_score(fmap="", importance_type="gain")
        return imp_feats

    def _select_important_features(self, imp_feats, top_num):
        sorted_feats = pd.Series(imp_feats).sort_values(ascending=False)
        print(
            f"        Features with importance >= {self.feature_importance_rate_limit}"
        )
        number_of_top_features = len(
            sorted_feats[
                sorted_feats / sorted_feats.max() >= self.feature_importance_rate_limit
            ]
        )
        if number_of_top_features > top_num:
            # 重要度が高い特徴量がtop_numより多い場合は、top_num個の特徴量を選択する
            print_feats = sorted_feats[:top_num].index.tolist()
        elif number_of_top_features > 1:
            print_feats = sorted_feats[
                sorted_feats / sorted_feats.max() >= self.feature_importance_rate_limit
            ].index.tolist()
        else:
            # 重要度が高い特徴量が1つの場合は、重要度が高い特徴量を1つ選択する
            print_feats = [sorted_feats.index[0]]
        return print_feats

    def _print_selected_features(self, print_feats):
        if len(print_feats) <= 30 and self.verbose:
            print("        Selected: %s" % print_feats)

    def _plot_feature_importances(self, bst_models):
        rows = int(len(bst_models) / 2 + 0.5)
        cols = 2
        fig, ax = plt.subplots(rows, cols)

        if rows == 1:
            ax = ax.reshape(-1, 1).T

        for i, model in enumerate(bst_models):
            row, col = divmod(i, cols)
            bst_booster = model.estimators_[0] if self.multi_label else model
            ax1 = xgb.plot_importance(
                bst_booster,
                height=0.8,
                show_values=False,
                importance_type="gain",
                max_num_features=10,
                ax=ax[row][col],
            )
            title = f"{'Multi_label: ' if self.multi_label else ''}Top 10 features{' for first label' if self.multi_label else ''}: round {i + 1}"
            ax1.set_title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    # Load and preprocess data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    print(X.shape)

    # Initialize FeatureSelection
    fs = RecursiveXGBoost(verbose=2, subset_number=5, feature_importance_rate_limit=0.1)

    # If you still want to remove columns with zero variance, do it before fitting fs
    # E.g., X = drop_zero_var_cols(X)

    # Transform X using the selected features
    X_selected = fs.fit_transform(X, y)
