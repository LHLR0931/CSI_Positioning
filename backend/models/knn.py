"""KNN 定位模型（阶段一基线）—— 传统 CSI 指纹匹配。

输入：任意形状的特征 [N, ...]，展平为 [N, D]。
封装 sklearn KNeighborsClassifier，提供 predict / predict_proba / save / load。

2026-10-12：加 standardize 选项（默认关，旧的阶段一 pilot 结果不受影响）。D1 起由 train_d1 使用
standardize=True，特征与样本图 GCN 完全相同（backend/models/sample_gcn.window_features），
两者的差别只剩"有没有图上的信息传递"。

自检运行：python -m backend.models.knn
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from backend import config


class KNNModel:
    name = "KNN"

    def __init__(self, k: int = 5, weights: str = "distance", metric: str = "minkowski",
                 n_jobs: int = 1, standardize: bool = False):
        self.k = k
        self.standardize = standardize
        self.mu = self.sd = None
        self.clf = KNeighborsClassifier(
            n_neighbors=k, weights=weights, metric=metric, n_jobs=n_jobs
        )

    def flatten(self, X: np.ndarray) -> np.ndarray:
        """[N, ...] -> [N, D]；standardize=True 时用训练集的均值/标准差标准化。"""
        F = np.asarray(X).reshape(X.shape[0], -1)
        return (F - self.mu) / self.sd if self.standardize else F

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNModel":
        if self.standardize:
            F = np.asarray(X).reshape(X.shape[0], -1)
            self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        self.clf.fit(self.flatten(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.flatten(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 [N, NUM_CLASSES]（补齐训练时未出现的类别列）。"""
        p = self.clf.predict_proba(self.flatten(X))
        full = np.zeros((X.shape[0], config.NUM_CLASSES), np.float64)
        full[:, self.clf.classes_] = p
        return full

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.standardize:      # 新格式：连同标准化参数一起存
            joblib.dump({"clf": self.clf, "mu": self.mu, "sd": self.sd}, path)
        else:                     # 旧格式不变（阶段一 pilot、API 仍按旧格式读取）
            joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path: Path) -> "KNNModel":
        m = cls()
        obj = joblib.load(path)
        if isinstance(obj, dict):
            m.clf, m.mu, m.sd, m.standardize = obj["clf"], obj["mu"], obj["sd"], True
        else:
            m.clf = obj
        return m


if __name__ == "__main__":
    from backend.training.prepare import load_stage1
    from backend.models.metrics import evaluate, format_report

    data, _ = load_stage1()
    Xtr, ytr = data["train"]
    Xte, yte = data["test"]
    model = KNNModel(k=5).fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)
    from backend.data.sessions import session_grid
    print(format_report(evaluate(yte, pred, proba, grid=session_grid("session_001")), "KNN"))
    print("KNN 自检通过。")
