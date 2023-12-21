import warnings
from math import pi

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def calculate_mi(
    X, y, task_type="classification", n_neighbors=5, discrete_features=False
):
    if task_type == "classification":
        mi = mutual_info_classif(
            X, y, n_neighbors=n_neighbors, discrete_features=discrete_features
        )
    else:
        mi = mutual_info_regression(
            X, y, n_neighbors=n_neighbors, discrete_features=discrete_features
        )
    mi /= np.max(mi)
    return mi


# select_features 関数を修正して、相互情報量も返すようにします
def select_features(X, y, task_type="classification", correlation_threshold=0.8):
    _X = X.copy()
    _y = y.copy()
    # 相互情報量を計算
    mi = calculate_mi(_X, _y, task_type=task_type)

    # 相関係数行列を計算
    corr_matrix = _X.corr()

    # 削除する列のインデックスを格納するリスト
    columns_to_drop = set()

    # 閾値を超える相関係数のペアを見つける
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                # 相関係数が高いペアから、相互情報量が低い特徴量を削除
                if mi[i] < mi[j]:
                    columns_to_drop.add(corr_matrix.columns[i])
                else:
                    columns_to_drop.add(corr_matrix.columns[j])

    selected_columns = _X.columns.difference(columns_to_drop)
    return selected_columns


def visualize_features(X, y, task_type="classification", correlation_threshold=0.8):
    sulov = SelectFeaturesBySulov(
        task_type=task_type, correlation_threshold=correlation_threshold
    )
    warnings.filterwarnings("ignore")
    selected_columns = sulov.fit_transform(X, y).columns

    mi_values = sulov.mi_
    mi_values = {feature: mi for feature, mi in zip(X.columns, mi_values)}
    corr_matrix = X.corr()

    sorted_features = sorted(mi_values.keys(), key=lambda x: mi_values[x], reverse=True)

    G = nx.Graph()
    angles = np.linspace(0, 2 * pi, len(sorted_features) + 1)
    pos = {}
    for i, feature in enumerate(sorted_features):
        if feature not in selected_columns:
            label = feature + " (removed)"
        else:
            label = feature
        pos[label] = np.array([np.cos(angles[i]), np.sin(angles[i])])

    for feature in sorted_features:
        label = feature if feature in selected_columns else feature + " (removed)"
        G.add_node(label, size=mi_values[feature])

    for i, feature1 in enumerate(sorted_features):
        for j, feature2 in enumerate(sorted_features):
            if (
                i < j
                and abs(corr_matrix.loc[feature1, feature2]) > correlation_threshold
            ):
                label1 = (
                    feature1
                    if feature1 in selected_columns
                    else feature1 + " (removed)"
                )
                label2 = (
                    feature2
                    if feature2 in selected_columns
                    else feature2 + " (removed)"
                )
                G.add_edge(
                    label1, label2, weight=abs(corr_matrix.loc[feature1, feature2])
                )

    plt.figure(figsize=(12, 12))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=[G.nodes[node]["size"] * 1000 for node in G.nodes],
        width=[G[u][v]["weight"] * 1 for u, v in G.edges],
    )

    plt.title("Feature Relationships")
    plt.show()


# カスタムトランスフォーマーの定義
class SelectFeaturesBySulov(BaseEstimator, TransformerMixin):
    def __init__(self, task_type="classification", correlation_threshold=0.8):
        self.task_type = task_type
        self.correlation_threshold = correlation_threshold

    def fit(self, X, y):
        print(f"X has {X.shape[1]} features")
        if self.task_type == "classification":
            mi = mutual_info_classif(X, y)
        else:
            mi = mutual_info_regression(X, y)
        mi /= np.max(mi)
        self.mi_ = mi

        # 相互情報量の大きさ順に特徴を並べ替える
        sorted_indices = np.argsort(mi)[::-1]  # 降順でインデックスをソート
        mi_sorted = mi[sorted_indices]
        self.sorted_features_ = X.columns[sorted_indices]
        _X = X[self.sorted_features_].copy()

        corr_matrix = _X.corr()
        self.columns_to_drop_ = set()

        for i in tqdm(range(len(corr_matrix.columns)), mininterval=10.0):
            if corr_matrix.columns[i] in self.columns_to_drop_:
                continue
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                    if mi_sorted[i] < mi_sorted[j]:
                        self.columns_to_drop_.add(corr_matrix.columns[i])
                    else:
                        self.columns_to_drop_.add(corr_matrix.columns[j])

        print(
            f"X has {X.shape[1] - len(self.columns_to_drop_)} features after selection"
        )
        return self

    def transform(self, X):
        _X = X.copy()
        return _X.drop(columns=self.columns_to_drop_)


if __name__ == "__main__":
    # Pipelineの作成
    pipeline = Pipeline(
        [
            (
                "feature_selection",
                SelectFeaturesBySulov(
                    task_type="classification", correlation_threshold=0.7
                ),
            ),
            # その他の処理 (例: 分類器)
        ]
    )
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    # Pipelineの実行
    pipeline.fit(X, y)
    X_transformed = pipeline.transform(X)
    # 可視化オプションが指定された場合の処理
    visualize = True  # 可視化を行う場合はTrueに設定
    if visualize:
        visualize_features(X, y)
