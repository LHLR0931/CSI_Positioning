"""data/<session>/ 新布局、网格随数据走、跳过点、子载波选择防泄漏（2026-10-07）。

用仓库里旧 pilot 的真实 pcap 拼出临时的 d1 / d3 目录，不依赖本机是否已有新数据。
"""
import json
import shutil

import numpy as np
import pytest

from backend import config
from backend.data import load_real_stage2, sessions

LEGACY_PCAP_DIR = config.SOURCE_DATA_DIR  # data/phase2_r*/label*_pi*.pcap（旧 pilot）


def _legacy_pcap(rnd: int, lab: int, pi: int):
    p = LEGACY_PCAP_DIR / f"phase2_r{rnd}" / f"label{lab}_pi{pi}.pcap"
    if not p.exists():
        pytest.skip(f"旧 pilot pcap 不在本机: {p}")
    return p


def _meta(rounds):
    return {
        "session": "x",
        "grid": {"Width": 5, "Height": 3, "Spacing": 0.8, "Path": "snake"},
        "devices": {},
        "pkts_per_pt": 1800,
        "rounds": rounds,
    }


@pytest.fixture
def tmp_roots(tmp_path, monkeypatch):
    src, raw = tmp_path / "data", tmp_path / "datasets_raw"
    src.mkdir()
    raw.mkdir()
    monkeypatch.setattr(config, "SOURCE_DATA_DIR", src)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "DATASET_ARTIFACTS_DIR", raw)
    return src, raw


def _make_session(src, session, rounds_labels, meta):
    """rounds_labels: {目标轮次: [label...]}，内容取自旧 pilot 的 r1/label0、label1。"""
    for rnd, labels in rounds_labels.items():
        d = src / session / f"phase2_r{rnd}"
        d.mkdir(parents=True)
        for lab in labels:
            for pi in (1, 2):
                shutil.copy(_legacy_pcap(1 + (rnd % 2), lab, pi), d / f"label{lab}_pi{pi}.pcap")
    if meta is not None:
        # PowerShell 5.1 的 Set-Content -Encoding UTF8 会写 BOM
        (src / session / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8-sig")


def test_new_layout_builds_with_session_grid_and_skips(tmp_roots):
    src, raw = tmp_roots
    meta = _meta([
        {"round": 1, "only_label": None, "skipped_labels": []},
        {"round": 2, "only_label": None, "skipped_labels": [1]},
    ])
    _make_session(src, "d1", {1: [0, 1], 2: [0, 1]}, meta)

    out = load_real_stage2.build("d1")
    d = np.load(out)
    assert float(d["grid_spacing_m"]) == 0.8 and int(d["grid_rows"]) == 3
    batch, label = d["batch"].astype(str), d["label"]
    assert set(label[batch == "r1"].tolist()) == {0, 1}
    assert set(label[batch == "r2"].tolist()) == {0}          # r2 label1 被 skip
    assert list(d["subcarrier_selection_batches"]) == ["r1", "r2"]
    raw64 = np.load(raw / "d1" / "static_raw64.npz")
    assert float(raw64["grid_spacing_m"]) == 0.8


def test_d1_heldout_rounds_never_select_subcarriers(tmp_roots):
    src, _ = tmp_roots
    _make_session(src, "d1", {1: [0], 7: [0], 8: [0], 9: [0]}, _meta([]))
    d = np.load(load_real_stage2.build("d1"))
    assert list(d["subcarrier_selection_batches"]) == ["r1"]   # r7 验证 / r8 标定 / r9 测试都不参与


def test_cross_day_session_requires_sc_from(tmp_roots):
    src, _ = tmp_roots
    _make_session(src, "d1", {1: [0]}, _meta([]))
    _make_session(src, "d3", {1: [0]}, _meta([]))
    with pytest.raises(RuntimeError, match="--sc-from"):
        load_real_stage2.build("d3")
    d1 = np.load(load_real_stage2.build("d1"))
    d3 = np.load(load_real_stage2.build("d3", sc_from="d1"))
    assert np.array_equal(d1["sc_idx"], d3["sc_idx"])
    assert all(b.startswith("d1:") for b in d3["subcarrier_selection_batches"].astype(str))


def test_unknown_session_never_falls_back_to_legacy_dir(tmp_roots):
    src, _ = tmp_roots
    (src / "phase2_r1").mkdir()          # 顶层旧布局存在，也不能被 d7 误读
    with pytest.raises(FileNotFoundError):
        load_real_stage2.build("d7")


def test_legacy_session_resolution_and_grid(tmp_roots):
    src, _ = tmp_roots
    (src / "phase2_r1").mkdir()
    (src / "phase2_r1" / "label0_pi1.pcap").write_bytes(b"")
    assert sessions.stage2_round_dirs("session_phase2_001") == [src / "phase2_r1"]
    assert sessions.session_grid("session_phase2_001").spacing_m == 1.0
    assert sessions.is_stage2("session_phase2_001")
    assert not sessions.is_stage2("session_002")


def test_skipped_point_restored_by_later_redo(tmp_roots):
    src, _ = tmp_roots
    (src / "d1").mkdir()
    (src / "d1" / "session_meta.json").write_text(json.dumps(_meta([
        {"round": 3, "only_label": None, "skipped_labels": [7, 9]},
        {"round": 3, "only_label": 7, "skipped_labels": []},      # -Only 7 -Force 重采成功
    ])), encoding="utf-8-sig")
    assert sessions.skipped_points("d1") == {(3, 9)}


def test_powershell_single_element_json_quirks(tmp_roots):
    src, _ = tmp_roots
    (src / "d1").mkdir()
    m = _meta(None)
    m["rounds"] = {"round": 2, "only_label": None, "skipped_labels": 4}   # 标量而非数组
    (src / "d1" / "session_meta.json").write_text(json.dumps(m), encoding="utf-8-sig")
    assert sessions.skipped_points("d1") == {(2, 4)}


def test_new_session_without_grid_meta_refuses_config_fallback(tmp_roots):
    src, _ = tmp_roots
    (src / "d1" / "phase2_r1").mkdir(parents=True)
    with pytest.raises(ValueError, match="session_meta"):
        sessions.session_grid("d1")
