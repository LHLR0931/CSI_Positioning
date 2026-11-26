"""实时后端的单元测试（不需要 Pi、不需要 torch）。

运行（项目根目录）：python -m pytest tests/test_live.py -q
需要 data/smoke/phase2_r1/ 下有冒烟测试的 pcap；没有就跳过。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from backend import config
from backend.data.load_real_stage2 import _read_pcap_split
from backend.live.linkbuf import LINKS, LinkBuffers
from backend.live.pcap_stream import PcapStreamParser

SMOKE = config.SOURCE_DATA_DIR / "smoke" / "phase2_r1"
PCAPS = sorted(SMOKE.glob("label*_pi*.pcap")) if SMOKE.exists() else []


@pytest.mark.skipif(not PCAPS, reason="没有 data/smoke 冒烟测试数据")
@pytest.mark.parametrize("chunk", [1, 7, 100, 4096, 10_000_000])
def test_stream_parser_matches_offline(chunk):
    """任意分块大小下，流式解析与离线 _read_pcap_split 逐帧完全一致（CSI 与时间戳）。"""
    for f in PCAPS:
        raw = f.read_bytes()
        ref = _read_pcap_split(io.BytesIO(raw))
        p = PcapStreamParser()
        got = {"apX": [], "apY": []}
        ts = {"apX": [], "apY": []}
        for s in range(0, len(raw), chunk):
            for t, ap, csi in p.feed(raw[s:s + chunk]):
                got[ap].append(csi)
                ts[ap].append(t)
        for ap in ("apX", "apY"):
            assert len(got[ap]) == len(ref[ap]["csi"]) > 300, (f.name, ap)
            np.testing.assert_array_equal(np.asarray(got[ap], np.complex64), ref[ap]["csi"])
            np.testing.assert_array_equal(np.asarray(ts[ap]), ref[ap]["timestamp"])


def test_window_needs_all_links_and_16_frames():
    b = LinkBuffers()
    now = 1000.0
    csi = np.ones(64, np.complex64)
    for i in range(16):
        for pi in ("piA", "piB"):
            b.add(pi, "apX", i * 0.1, csi * (i + 1), now=now)
    assert b.window(now=now) is None                     # apY 两条链路没有数据
    for i in range(16):
        for pi in ("piA", "piB"):
            b.add(pi, "apY", i * 0.1, csi, now=now)
    w = b.window(now=now)
    assert w is not None
    amp, phase, end, span, totals = w
    assert amp.shape == phase.shape == (16, 4, 64)
    assert amp[-1, LINKS.index("piA_apX"), 0] == 16       # 最近一帧在最后
    assert abs(span - 1.5) < 1e-9 and end == now
    assert b.window(now=now + 2.5) is None               # 超过 2 秒没有新帧 → 不出窗口
    h = b.health(now=now + 2.5)
    assert all(not v["online"] for v in h.values())
