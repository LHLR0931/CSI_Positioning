"""统一评估指标（对应方案文档第 10 节）。

所有模型/阶段共用这一套指标，保证结果可比：
  分类准确率、Top-3 准确率、平均/90 分位定位误差、每点准确率、误差 CDF、混淆矩阵。
  中位数仍由 error_stats 计算但不进任何报告：离散格点下它坍缩到格值，无分辨力。

纯 numpy，无框架依赖。

网格（间距 / 行列）必须由调用方显式给出：``grid=sessions.session_grid(session)``。
2026-10-07 前这里默认读全局 ``config.GRID_SPACING_M``；09-12 config 改成 0.8 m 后，
1.0 m 的旧数据被整体乘了 0.8（报告里 0.827 m 实为 1.034 m）。现在不给网格直接报错。

自检运行：python -m backend.models.metrics
"""

from __future__ import annotations

import numpy as np

from backend import config
from backend.data.sessions import Grid


def _spacing(grid: Grid | None, spacing: float | None) -> float:
    if spacing is not None:
        return float(spacing)
    if grid is not None:
        return float(grid.spacing_m)
    raise ValueError(
        "定位误差必须显式给出网格：grid=sessions.session_grid(session)（或 spacing=...）。"
        "不再默认读全局 config.GRID_SPACING_M —— 它与数据实际间距不一致时会把误差整体缩放。"
    )


def _cols(grid: Grid | None) -> int:
    return grid.cols if grid is not None else config.GRID_COLS


def _xy(labels, grid: Grid | None):
    labels = np.asarray(labels)
    c = _cols(grid)
    return labels % c, labels // c


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def top_k_accuracy(y_true, proba, k: int = 3) -> float:
    """proba: [N, NUM_CLASSES]。"""
    proba = np.asarray(proba)
    topk = np.argsort(-proba, axis=1)[:, :k]
    y_true = np.asarray(y_true)[:, None]
    return float(np.mean((topk == y_true).any(axis=1)))


def positioning_errors(y_true, y_pred, spacing: float | None = None, *,
                       grid: Grid | None = None) -> np.ndarray:
    """预测类别与真实类别映射为坐标后的欧氏距离(米)。必须给 grid 或 spacing。"""
    s = _spacing(grid, spacing)
    tx, ty = _xy(y_true, grid)
    px, py = _xy(y_pred, grid)
    return np.hypot(tx - px, ty - py).astype(np.float64) * s


def error_stats(errors: np.ndarray) -> dict:
    e = np.asarray(errors, np.float64)
    return {
        "mean_error_m": float(e.mean()),
        "median_error_m": float(np.median(e)),
        "p90_error_m": float(np.percentile(e, 90)),
        "max_error_m": float(e.max()),
    }


def cdf_points(errors: np.ndarray):
    """返回 (排序误差, 对应累计概率)，可直接画 CDF。"""
    e = np.sort(np.asarray(errors, np.float64))
    c = np.arange(1, len(e) + 1) / len(e)
    return e, c


def confusion_matrix(y_true, y_pred, num_classes: int | None = None, *,
                     grid: Grid | None = None) -> np.ndarray:
    if num_classes is None:
        num_classes = grid.num_classes if grid is not None else config.NUM_CLASSES
    m = np.zeros((num_classes, num_classes), np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        m[int(t), int(p)] += 1
    return m


def evaluate(y_true, y_pred, proba=None, *, grid: Grid | None = None,
             spacing: float | None = None) -> dict:
    """汇总全部指标为一个 dict。必须给 grid 或 spacing。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = positioning_errors(y_true, y_pred, spacing, grid=grid)
    out = {"accuracy": accuracy(y_true, y_pred), **error_stats(err),
           "n": int(len(y_true))}
    if proba is not None:
        out["top3_accuracy"] = top_k_accuracy(y_true, proba, k=3)
    return out


def positioning_errors_xy(pred_xy, true_xy, spacing: float | None = None, *,
                          grid: Grid | None = None) -> np.ndarray:
    """连续坐标回归的定位误差(米)。pred_xy/true_xy: [N,2]，单位为格。必须给 grid 或 spacing。"""
    s = _spacing(grid, spacing)
    return np.linalg.norm(np.asarray(pred_xy) - np.asarray(true_xy), axis=1) * s


def evaluate_xy(pred_xy, true_xy, *, grid: Grid | None = None,
                spacing: float | None = None) -> dict:
    """回归输出的评估汇总。必须给 grid 或 spacing。"""
    err = positioning_errors_xy(pred_xy, true_xy, spacing, grid=grid)
    return {**error_stats(err), "n": int(len(true_xy))}


def snap_to_label(pred_xy, grid: Grid) -> np.ndarray:
    """连续坐标预测 snap 到最近格点，返回离散 label（越界先 clip 到**该数据集**网格范围再取整）。

    行数必须来自数据集：旧数据 2 排，用全局 config 的 3 排去 clip 会允许预测落到不存在的第三排。
    """
    if grid is None:
        raise ValueError("snap_to_label 必须给 grid（行列数决定 clip 范围）")
    pred_xy = np.asarray(pred_xy)
    x = np.clip(np.round(pred_xy[:, 0]), 0, grid.cols - 1).astype(int)
    y = np.clip(np.round(pred_xy[:, 1]), 0, grid.rows - 1).astype(int)
    return y * grid.cols + x


def y_mismatch_rate(y_true_label, y_pred_label, *, grid: Grid | None = None) -> float:
    """y 维（排）错分率：snap 后预测的 y 与真实 y 不一致的样本比例。

    2 排时它就是"左右对称混淆率"。3 排之后单靠这个比率不够：Δy=1 与 Δy=2
    都计为一次错分，但物理代价差一倍（0.8 m vs 1.6 m），所以必须与
    ``mean_abs_dy`` 一起报告。
    """
    _, ty = _xy(y_true_label, grid)
    _, py = _xy(y_pred_label, grid)
    return float(np.mean(ty != py))


def mean_abs_dy(y_true_label, y_pred_label, *, grid: Grid | None = None) -> float:
    """y 维平均错位格数 E[|Δy|]。3 排网格下与 ``y_mismatch_rate`` 配套使用：
    比率说"错了多少次"，这个说"每次错多远"。乘该数据集的 grid.spacing_m 即为米。
    """
    _, ty = _xy(y_true_label, grid)
    _, py = _xy(y_pred_label, grid)
    return float(np.mean(np.abs(ty.astype(np.int64) - py.astype(np.int64))))


def per_point_accuracy(y_true_label, y_pred_label,
                       num_classes: int | None = None, *, grid: Grid | None = None) -> dict:
    """每个格点的绝对准确率 {label: acc}，外加样本数，供逐点热力图与表格使用。

    label 未出现在测试集里时 acc 记为 None（而不是 0），避免把"没测到"说成"全错"。
    """
    if num_classes is None:
        num_classes = grid.num_classes if grid is not None else config.NUM_CLASSES
    y_true = np.asarray(y_true_label)
    y_pred = np.asarray(y_pred_label)
    out = {}
    for c in range(num_classes):
        m = y_true == c
        n = int(m.sum())
        out[int(c)] = {"n": n,
                       "accuracy": (float(np.mean(y_pred[m] == c)) if n else None)}
    return out


def format_report(metrics: dict, title: str = "") -> str:
    """Format the core evaluation metrics as an English console report.

    Top-3 accuracy, the 90th percentile, and the maximum error are still
    calculated by ``evaluate`` and stored in the metrics dictionary.
    """
    heading = f"Evaluation Report: {title}" if title else "Evaluation Report"
    lines = [f"--- {heading} ---"]
    lines.append(f"  Samples                   : {metrics.get('n', '?')}")
    if "accuracy" in metrics:
        lines.append(f"  Accuracy                  : {metrics['accuracy']:.1%}")
    lines.append(f"  Mean localization error   : {metrics['mean_error_m']:.3f} m")
    if "p90_error_m" in metrics:
        lines.append(f"  90th-percentile error     : {metrics['p90_error_m']:.3f} m")
    return "\n".join(lines)


def core_metrics(metrics: dict, errors=None) -> dict:
    """从完整 metrics 里挑出核心结果 (+ 可选的 CDF 数据)，用于精简落盘。

    不含中位数：离散格点上误差只能取 {0, 1, √2, 2, ...}×间距 这样的格值，
    中位数会坍缩到其中一个格值上，分辨不出模型差异。用 p90 代替。
    """
    out = {
        "accuracy": metrics.get("accuracy"),
        "mean_error_m": metrics["mean_error_m"],
        "p90_error_m": metrics.get("p90_error_m"),
        "n": metrics.get("n"),
    }
    if errors is not None:
        e, c = cdf_points(errors)
        out["cdf_error_m"] = e.tolist()
        out["cdf_p"] = c.tolist()
    return out


def plot_cdf(errors, path, title: str = "Localization error CDF"):
    """Plot the error CDF and save it as a PNG file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot_cdf] Plot skipped because matplotlib is unavailable: {exc}. "
              "CDF data is still available in the JSON metrics file.")
        return None
    from pathlib import Path as _P

    e, c = cdf_points(errors)
    path = _P(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(e, c, lw=2)
    ax.set_xlabel("Localization error (m)")
    ax.set_ylabel("CDF probability")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot_cdf] Saved CDF plot -> {path}")
    return path


if __name__ == "__main__":
    from backend.data.sessions import config_grid
    G = config_grid()   # 模拟数据自检，没有真实数据集可读
    rng = np.random.default_rng(0)
    N = 500
    y = rng.integers(0, config.NUM_CLASSES, N)
    # 构造一个「大多数正确、少数偏一格」的预测，验证指标合理
    y_pred = y.copy()
    flip = rng.random(N) < 0.2
    y_pred[flip] = np.clip(y_pred[flip] + 1, 0, config.NUM_CLASSES - 1)
    proba = np.zeros((N, config.NUM_CLASSES))
    proba[np.arange(N), y_pred] = 0.7
    proba[np.arange(N), y] += 0.3
    m = evaluate(y, y_pred, proba, grid=G)
    print(format_report(m, "Self-test"))
    e, c = cdf_points(positioning_errors(y, y_pred, grid=G))
    print(f"  CDF: error range={e[0]:.1f}..{e[-1]:.1f} m, "
          f"50th percentile={e[np.searchsorted(c, 0.5)]:.2f} m")
    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion matrix shape={cm.shape}, "
          f"diagonal ratio={np.trace(cm)/cm.sum():.1%}")
    assert 0.75 < m["accuracy"] < 0.85
    print("Metrics module self-test passed.")
