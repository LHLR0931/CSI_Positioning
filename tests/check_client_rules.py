"""按 Unity 客户端的校验规则（unity-app/Assets/CSI/Integration/CsiContracts.cs 的
CsiValidation.Meta / Prediction）检查后端的真实响应。不符合的预测会被前端直接丢弃，所以上线前必须全过。

用法（后端在运行时）：python tests/check_client_rules.py http://127.0.0.1:8765 [秒数，默认 20]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        body = r.read()
        return r.status, (json.loads(body) if body else None)


def _time(s) -> datetime:
    if not isinstance(s, str) or "T" not in s:
        raise ValueError(f"时间不是 ISO 格式：{s!r}")
    if not (s.endswith("Z") or (len(s) >= 6 and s[-6] in "+-" and s[-3] == ":")):
        raise ValueError(f"时间没有时区：{s!r}")
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def check_meta(m: dict) -> None:
    g = m["grid"]
    assert g["cols"] == 5 and g["rows"] == 3 and abs(g["spacing_m"] - 0.8) <= 0.001, "网格不是 5×3 / 0.8 m"
    assert len(m["cells"]) == 15
    seen = set()
    for c in m["cells"]:
        l = c["label"]
        assert 0 <= l <= 14 and l not in seen
        seen.add(l)
        assert c["x"] == l % 5 and c["y"] == l // 5
        assert abs(c["x_m"] - c["x"] * 0.8) <= 0.005 and abs(c["y_m"] - c["y"] * 0.8) <= 0.005


def check_status(s: dict) -> None:
    assert s["state"] in ("offline", "calibrating", "training", "ready", "degraded")
    assert s["progress"] is None or 0 <= s["progress"] <= 1
    _time(s["timestamp"])
    for l in ("piA_apX", "piA_apY", "piB_apX", "piB_apY"):
        v = s["links"][l]
        assert isinstance(v["online"], bool) and isinstance(v["fps"], (int, float))
        assert isinstance(v["last_frame_age_s"], (int, float))


def check_prediction(p: dict) -> None:
    l = p["label"]
    assert 0 <= l <= 14 and p["seq"] >= 0
    ts = _time(p["timestamp"])
    assert p["x"] == l % 5 and p["y"] == l // 5
    assert abs(p["x_m"] - p["x"] * 0.8) <= 0.005 and abs(p["y_m"] - p["y"] * 0.8) <= 0.005
    probs = p["probs"]
    assert len(probs) == 15 and 0 <= p["confidence"] <= 1
    assert all(0 <= v <= 1 for v in probs)
    assert abs(sum(probs) - 1) <= 0.025, f"概率和 {sum(probs)}"
    assert abs(p["confidence"] - probs[l]) <= 0.005, "confidence ≠ probs[label]"
    assert probs[l] >= max(probs) - 1e-5, "label 不是概率最大的格点"
    e = p["expected_m"]
    assert -0.005 <= e["x"] <= 3.205 and -0.005 <= e["y"] <= 1.605
    ex = sum(v * (i % 5) * 0.8 for i, v in enumerate(probs))
    ey = sum(v * (i // 5) * 0.8 for i, v in enumerate(probs))
    assert abs(e["x"] - ex) <= 0.03 and abs(e["y"] - ey) <= 0.03, "expected_m 与概率加权坐标不一致"
    assert p["state"] in ("ready", "degraded")
    w = p["window"]
    start, end = _time(w["start"]), _time(w["end"])
    assert start <= end and end == ts and w["frames"] >= 1, "window 时间不对（end 必须等于 timestamp）"


def main() -> None:
    base = sys.argv[1].rstrip("/")
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    code, meta = _get(base, "/api/meta")
    check_meta(meta)
    print("meta ✓")
    n_pred, n_204, last_seq, states = 0, 0, -1, set()
    t_end = time.time() + secs
    while time.time() < t_end:
        _, st = _get(base, "/api/status")
        check_status(st)
        states.add(st["state"])
        code, p = _get(base, "/api/predict/latest")
        if code == 204:
            n_204 += 1
        else:
            check_prediction(p)
            if p["seq"] != last_seq:
                assert p["seq"] > last_seq, "seq 没有递增"
                last_seq = p["seq"]
                n_pred += 1
        time.sleep(0.5)
    print(f"status ✓（出现过的状态：{sorted(states)}）")
    print(f"prediction ✓ 收到 {n_pred} 条不同的预测，{n_204} 次 204；全部通过前端校验规则")


if __name__ == "__main__":
    main()
