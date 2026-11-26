#!/usr/bin/env python3
"""Synthetic HTTP integration server for the CSI v0.1 draft contract.

This does not read CSI, train models, or implement the teammate's real backend.
Default responses advertise a LOCAL OPTIONAL four-model extension. --v01-only
returns one unlabelled v0.1 prediction instead. HTTP polling is supported; this
stdlib fixture server deliberately does not implement WebSocket.

Run from the project root:
  python tools/csi_mock_server.py --host 127.0.0.1 --port 8000 --enable-control
  python -m unittest tests.test_csi_mock_server -v

Mock-only POST /control accepts state, fault, progress, offline_links, freeze,
advance, restart, and multi_model. It is disabled unless --enable-control is set.
Example: {"state":"calibrating","progress":0.4} then {"state":"ready"}.
Faults: none, http_error, malformed_json, invalid_probs, nonfinite,
out_of_order, stale. These controls are NOT requests for real backend APIs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


MODEL_IDS = ("KNN", "GCN", "ConcatMLP", "ConcatMLP-mask")
LINK_IDS = ("piA_apX", "piA_apY", "piB_apX", "piB_apY")
STATES = ("offline", "calibrating", "training", "ready", "degraded")
FAULTS = ("none", "http_error", "malformed_json", "invalid_probs", "nonfinite",
          "out_of_order", "stale")
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "integration"
PROVENANCE = "SYNTHETIC frontend integration fixture; no CSI or trained model inference."


def iso_time(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_meta(session_id: str = "synthetic-replay-v1", multi_model: bool = True) -> dict:
    devices = [("piA", "pi", -0.9, 2.0, 1.1), ("piB", "pi", 2.65, 2.2, 0.6),
               ("apX", "ap", 0.75, -0.7, 0.75), ("apY", "ap", 4.1, -0.3, 1.35)]
    result = {
        "contract_version": "0.1", "source": "mock", "synthetic": True,
        "provenance": PROVENANCE,
        "grid": {"cols": 5, "rows": 3, "spacing_m": 0.8},
        "cells": [{"label": i, "x": i % 5, "y": i // 5,
                   "x_m": round((i % 5) * 0.8, 2), "y_m": round((i // 5) * 0.8, 2)}
                  for i in range(15)],
        "devices": [{"id": i, "kind": k, "x_m": x, "y_m": y, "h_m": h}
                    for i, k, x, y, h in devices],
        "links": [{"id": i, "pi": i.split("_")[0], "ap": i.split("_")[1]}
                  for i in LINK_IDS],
        "window_frames": 16, "frame_rate_hz": 10,
        "mock_transport": "http-polling", "mock_websocket_supported": False,
    }
    if multi_model:
        result.update(session_id=session_id, supported_models=list(MODEL_IDS),
                      default_model_id="ConcatMLP-mask",
                      extensions=["local.mock.four-model.v1", "local.mock.replay.v1"])
    return result


def load_scenario(path: Path | None = None) -> dict:
    scenario = json.loads((path or FIXTURE_DIR / "synthetic_scenario.json").read_text(encoding="utf-8"))
    if scenario.get("source") != "mock" or not scenario.get("frames"):
        raise ValueError("Scenario must contain explicitly synthetic frames")
    for frame in scenario["frames"]:
        for label in [frame["ground_truth_label"], *[frame["predicted_labels"][m] for m in MODEL_IDS]]:
            if type(label) is not int or not 0 <= label < 15:
                raise ValueError("Fixture labels must be integers in 0..14")
    return scenario


def make_prediction(label: int, model_id: str, seq: int, timestamp: datetime,
                    session_id: str, state: str = "ready", multi_model: bool = True,
                    offline_links: tuple[str, ...] | list[str] = ()) -> dict:
    # Probability shape is synthetic and deliberately NOT a calibrated accuracy.
    probs = [0.02] * 15
    probs[label] = 0.72
    result = {
        "type": "prediction", "contract_version": "0.1", "source": "mock",
        "synthetic": True, "seq": seq, "timestamp": iso_time(timestamp),
        "window": {"start": iso_time(timestamp - timedelta(seconds=1.6)),
                   "end": iso_time(timestamp), "frames": 16},
        "label": label, "x": label % 5, "y": label // 5,
        "x_m": round(label % 5 * 0.8, 2), "y_m": round(label // 5 * 0.8, 2),
        "confidence": probs[label], "probs": probs,
        "expected_m": {
            "x": round(sum(p * (i % 5) * 0.8 for i, p in enumerate(probs)), 6),
            "y": round(sum(p * (i // 5) * 0.8 for i, p in enumerate(probs)), 6)},
        "links_used": [i for i in LINK_IDS if i not in offline_links], "state": state,
    }
    if multi_model:
        result.update(model_id=model_id, window_id=f"window-{seq:08d}", session_id=session_id)
    return result


def make_replay(scenario: dict | None = None) -> dict:
    scenario = scenario or load_scenario()
    session_id = "synthetic-replay-v1"
    start = datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)
    predictions = []
    for seq, frame in enumerate(scenario["frames"]):
        stamp = start + timedelta(seconds=seq * 0.4)
        for model_id in MODEL_IDS:
            row = make_prediction(frame["predicted_labels"][model_id], model_id, seq, stamp, session_id)
            # Truth comes from its own predeclared fixture label, never a prediction.
            row.update(ground_truth_label=frame["ground_truth_label"], truth_source="fixture_label")
            predictions.append(row)
    return {"source": "mock", "synthetic": True, "provenance": PROVENANCE,
            "session_id": session_id, "meta": make_meta(session_id), "predictions": predictions}


class MockState:
    def __init__(self, interval: float = 0.4, multi_model: bool = True,
                 state: str = "ready", fault: str = "none", scenario: dict | None = None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        if state not in STATES or fault not in FAULTS:
            raise ValueError("Unknown initial state or fault")
        self.lock = threading.RLock()
        self.interval, self.multi_model = interval, multi_model
        self.state, self.fault = state, fault
        self.scenario = scenario or load_scenario()
        self.progress = None
        self.offline_links = ["piB_apY"] if state == "degraded" else []
        self.frozen = False
        self.session_id = ""
        self.restart()

    def restart(self):
        self.session_id = "mock-" + uuid.uuid4().hex[:12]
        self.anchor = time.monotonic()
        self.anchor_utc = datetime.now(timezone.utc)
        self.base_seq = 0

    def current_seq(self) -> int:
        return self.base_seq + (0 if self.frozen else int((time.monotonic() - self.anchor) / self.interval))

    def apply_control(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Control body must be a JSON object")
        allowed = {"state", "fault", "progress", "offline_links", "freeze", "advance", "restart", "multi_model"}
        if set(data) - allowed:
            raise ValueError("Unknown control keys: " + ", ".join(sorted(set(data) - allowed)))
        if "state" in data and data["state"] not in STATES:
            raise ValueError("Unknown state")
        if "fault" in data and data["fault"] not in FAULTS:
            raise ValueError("Unknown fault")
        for key in ("freeze", "restart", "multi_model"):
            if key in data and type(data[key]) is not bool:
                raise ValueError(key + " must be boolean")
        if "advance" in data and (type(data["advance"]) is not int or data["advance"] < 0):
            raise ValueError("advance must be a nonnegative integer")
        if "progress" in data and data["progress"] is not None:
            p = data["progress"]
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("progress must be null or a number in 0..1")
        if "offline_links" in data:
            links = data["offline_links"]
            if not isinstance(links, list) or any(i not in LINK_IDS for i in links):
                raise ValueError("offline_links must be a list of known link IDs")
        with self.lock:
            if data.get("restart"):
                self.restart()
            if "freeze" in data and data["freeze"] != self.frozen:
                self.base_seq = self.current_seq()
                self.anchor = time.monotonic()
                self.frozen = data["freeze"]
            self.base_seq += data.get("advance", 0)
            for name in ("state", "fault", "progress", "multi_model", "offline_links"):
                if name in data:
                    setattr(self, name, copy.deepcopy(data[name]))
            if self.state == "degraded" and "offline_links" not in data and not self.offline_links:
                self.offline_links = ["piB_apY"]
            if self.state == "ready" and "offline_links" not in data:
                self.offline_links = []
            return {"source": "mock", "session_id": self.session_id, "state": self.state,
                    "fault": self.fault, "seq": self.current_seq(), "freeze": self.frozen,
                    "multi_model": self.multi_model}

    def status(self) -> dict:
        with self.lock:
            now = datetime.now(timezone.utc)
            links = {}
            for i in LINK_IDS:
                online = self.state != "offline" and i not in self.offline_links
                links[i] = {"online": online, "fps": 10.0 if online else 0.0,
                            "last_frame_age_s": 0.05 if online else 3.0}
            return {"type": "status", "contract_version": "0.1", "source": "mock",
                    "synthetic": True, "session_id": self.session_id, "timestamp": iso_time(now),
                    "state": self.state, "progress": self.progress if self.state in ("calibrating", "training") else None,
                    "calibrated_at": iso_time(self.anchor_utc) if self.state in ("ready", "degraded") else None,
                    "links": links, "message": "MOCK: " + self.state}

    def latest(self, model_id: str | None = None) -> tuple[int, object]:
        with self.lock:
            if model_id is not None and model_id not in MODEL_IDS:
                return 400, {"source": "mock", "error": "Unknown model_id", "supported_models": list(MODEL_IDS)}
            if self.fault == "http_error":
                return 503, {"source": "mock", "error": "Deliberate mock HTTP failure"}
            if self.state not in ("ready", "degraded"):
                return 204, None
            seq = self.current_seq()
            if self.fault == "out_of_order":
                seq = max(0, seq - 2)
            stamp = self.anchor_utc + timedelta(seconds=seq * self.interval)
            if self.fault == "stale":
                stamp -= timedelta(seconds=30)
            frame = self.scenario["frames"][seq % len(self.scenario["frames"])]
            models = [model_id] if model_id else (MODEL_IDS if self.multi_model else ["ConcatMLP-mask"])
            rows = [make_prediction(frame["predicted_labels"][m], m, seq, stamp, self.session_id,
                                    self.state, self.multi_model, self.offline_links) for m in models]
            if self.fault in ("invalid_probs", "nonfinite"):
                for row in rows:
                    row["probs"][0] = 2.0 if self.fault == "invalid_probs" else float("nan")
            if model_id or not self.multi_model:
                return 200, rows[0]
            return 200, {"type": "prediction_batch", "contract_version": "0.1", "source": "mock",
                         "synthetic": True, "session_id": self.session_id,
                         "window_id": f"window-{seq:08d}", "seq": seq,
                         "timestamp": iso_time(stamp), "predictions": rows}


class MockHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, mock_state: MockState | None = None, enable_control: bool = False):
        self.mock_state = mock_state or MockState()
        self.enable_control = enable_control
        super().__init__(address, MockHandler)


class MockHandler(BaseHTTPRequestHandler):
    server_version = "CSI-SyntheticMock/1.0"

    def _send(self, code: int, payload=None, raw: bytes | None = None):
        if code == 204:
            body = b""
        else:
            body = raw if raw is not None else json.dumps(payload, ensure_ascii=False, allow_nan=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-CSI-Source", "mock")
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        parsed = urlsplit(self.path)
        state = self.server.mock_state
        if parsed.path == "/api/meta":
            with state.lock:
                self._send(200, make_meta(state.session_id, state.multi_model))
        elif parsed.path == "/api/status":
            self._send(200, state.status())
        elif parsed.path == "/api/predict/latest":
            values = parse_qs(parsed.query)
            model_id = values.get("model_id", [None])[0]
            with state.lock:
                code, result = state.latest(model_id)
                if code == 200 and state.fault == "malformed_json":
                    self._send(200, raw=b'{"source":"mock","type":"prediction",BROKEN')
                else:
                    self._send(code, result)
        elif parsed.path in ("/api/replay", "/replay.json"):
            self._send(200, make_replay(state.scenario))
        elif parsed.path == "/ws/predictions":
            self._send(501, {"source": "mock", "error": "Mock supports HTTP polling only; no WebSocket implementation"})
        else:
            self._send(404, {"source": "mock", "error": "Unknown mock endpoint"})

    def do_POST(self):
        if urlsplit(self.path).path != "/control" or not self.server.enable_control:
            self._send(404, {"source": "mock", "error": "Mock controls disabled or endpoint unknown"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Control body must be 1..65536 bytes")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send(200, self.server.mock_state.apply_control(data))
        except (ValueError, UnicodeDecodeError, TypeError) as exc:
            self._send(400, {"source": "mock", "error": str(exc)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 explicitly for a LAN device")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval", type=float, default=0.4, help="Synthetic window interval in seconds")
    parser.add_argument("--v01-only", action="store_true", help="Single unlabelled v0.1 row, no advertised four-model capability")
    parser.add_argument("--enable-control", action="store_true", help="Enable MOCK-ONLY POST /control fault/state test endpoint")
    parser.add_argument("--state", choices=STATES, default="ready")
    parser.add_argument("--fault", choices=FAULTS, default="none")
    parser.add_argument("--scenario", type=Path, help="Alternative explicitly synthetic scenario JSON")
    args = parser.parse_args(argv)
    try:
        state = MockState(args.interval, not args.v01_only, args.state, args.fault, load_scenario(args.scenario))
        server = MockHTTPServer((args.host, args.port), state, args.enable_control)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"MOCK ONLY: {PROVENANCE}\nHTTP polling at http://{args.host}:{server.server_port}\n"
          f"Four-model extension: {state.multi_model}; mock controls: {args.enable_control}\n"
          "Replay: /api/replay ; Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nMock server stopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
