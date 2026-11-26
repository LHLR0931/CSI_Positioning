"""评估指标的单元测试。"""

import numpy as np
import pytest

from backend.data.sessions import Grid, session_grid
from backend.models import metrics

G_NEW = Grid(cols=5, rows=3, spacing_m=0.8)     # D1 起的新场地
G_OLD = session_grid("session_phase2_001")      # 旧 pilot：5x2 @ 1.0 m


def test_accuracy_and_positioning_error():
    y = np.array([0, 1, 2])
    assert metrics.accuracy(y, y) == 1.0
    # 预测差一格 (label 0 -> label 1 = x 差 1)，误差应为 1 格 * 该数据集的间距
    err = metrics.positioning_errors(np.array([0]), np.array([1]), grid=G_NEW)
    assert abs(err[0] - 0.8) < 1e-9


def test_top_k_accuracy():
    proba = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
    true = np.array([2, 0])
    # top1: s0 argmax=1(miss), s1 argmax=0(hit) -> 0.5
    assert metrics.top_k_accuracy(true, proba, k=1) == 0.5
    # top2: s0 {1,2} 含2(hit), s1 {0,1} 含0(hit) -> 1.0
    assert metrics.top_k_accuracy(true, proba, k=2) == 1.0


def test_regression_error_xy():
    pred = np.array([[0.0, 0.0], [3.0, 0.0]])
    true = np.array([[0.0, 0.0], [0.0, 0.0]])
    e = metrics.positioning_errors_xy(pred, true, grid=G_NEW)
    assert abs(e[0]) < 1e-9 and abs(e[1] - 3 * 0.8) < 1e-9


def test_confusion_matrix_shape():
    y = np.array([0, 1, G_NEW.num_classes - 1])
    cm = metrics.confusion_matrix(y, y, grid=G_NEW)
    assert cm.shape == (15, 15)
    assert np.trace(cm) == 3
    assert metrics.confusion_matrix(np.array([0, 9]), np.array([0, 9]), grid=G_OLD).shape == (10, 10)


# ---- 2026-10-07：尺度 bug 的回归测试 ----

def test_legacy_data_uses_its_own_1m_spacing():
    """旧数据 1 格 = 1.0 m。09-12 的报告把它按 0.8 m 算，ConcatMLP 1.034 m 被报成 0.827 m。"""
    pred = np.array([[1.0, 0.0]])
    true = np.array([[0.0, 0.0]])
    assert metrics.positioning_errors_xy(pred, true, grid=G_OLD)[0] == pytest.approx(1.0)
    assert metrics.evaluate_xy(pred, true, grid=G_OLD)["mean_error_m"] == pytest.approx(1.0)


def test_missing_grid_is_an_error_not_a_silent_config_default():
    with pytest.raises(ValueError, match="grid"):
        metrics.positioning_errors(np.array([0]), np.array([1]))
    with pytest.raises(ValueError, match="grid"):
        metrics.evaluate_xy(np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="grid"):
        metrics.snap_to_label(np.zeros((1, 2)), None)


def test_snap_clips_to_dataset_rows():
    """旧数据只有 2 排：预测 y=1.9 必须 clip 到第 2 排（y=1），不能落到不存在的第 3 排。"""
    pred = np.array([[2.0, 1.9]])
    assert metrics.snap_to_label(pred, G_OLD)[0] == 1 * 5 + 2
    assert metrics.snap_to_label(pred, G_NEW)[0] == 2 * 5 + 2
