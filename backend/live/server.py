"""实时定位后端：两台 Pi 实时取数 → F-MLP → HTTP 接口（约定 backend/API_CONTRACT.md v0.1）。

用法（项目根目录，csi 环境）：
    python -X utf8 -m backend.live.server --session d14
        session 已有训练好的模型 → 直接进入 ready；
        还没有 → 进入 calibrating，读 phase2_capture.ps1 的记录报告标定进度；
                  整轮标定采完后自动解析 + 训练（training），完成后加载模型进入 ready。
    测试（不连 Pi，回放已采 pcap）：
    python -X utf8 -m backend.live.server --session d14 --replay data\\smoke\\phase2_r1

接口（只用标准库，不需要装 fastapi）：
    GET /api/meta            网格、格点、设备、链路
    GET /api/status          状态与四条链路是否在线
    GET /api/predict/latest  最近一次预测；还没有预测时 204
    /ws/predictions          本版未实现（501）；客户端使用 HTTP 轮询

在约定 v0.1 之外多给的字段（前端可以忽略）：
    source（"live" / "replay"）、session_id（每次启动不同，前端据此知道后端重启过）、
    预测里的 model_id（"F-MLP"）。

约定 v0.1 第 8 节待定事项，本版的定法：
    推送频率   每 0.4 秒最多出一条预测（16 帧窗口，每条链路约 4 帧新数据）
    断链       任一条链路超过 2 秒没有数据 → state = degraded，**暂停出预测**
               （F-MLP 训练时四条链路都在，缺一条时的输出没有依据，所以不出）
    标定进度   progress = 已采格点数 / 15，message 写 "Calibrating: 7 / 15 points"
    IP         后端监听 0.0.0.0:8765，客户端填写后端电脑在同一局域网里的 IP
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from backend import config
from backend.data import sessions
from backend.live import fmlp
from backend.live.linkbuf import LINKS, LinkBuffers
from backend.live.train import calib_progress

CONTRACT_VERSION = "0.1"
STEP_S = 0.4


def iso(t: float) -> str:
    """带时区的 ISO 8601，毫秒精度。"""
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


class LiveBackend:
    def __init__(self, session: str, calib_round: int, auto_train: bool, source_kind: str,
                 log_dir: Path):
        self.session, self.calib_round = session, calib_round
        self.auto_train, self.source_kind = auto_train, source_kind
        self.session_id = f"{session}-{uuid.uuid4().hex[:8]}"
        self.buf = LinkBuffers()
        self.lock = threading.Lock()
        self.model: fmlp.LiveFMLP | None = None
        self.model_state = "calibrating"          # calibrating / training / ready / failed
        self.calib = {"n_done": 0, "complete": False, "calibrated_at": None}
        self.latest: dict | None = None
        self.seq = 0
        self.paused_reason: str | None = None
        self._stop = threading.Event()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._log_fh = open(self.log_path, "a", encoding="utf-8")
        self.grid = sessions.session_grid(session if (sessions.session_dir(session) / "session_meta.json").exists()
                                          else "d1")
        self.devices = (sessions.read_session_meta(session) or sessions.read_session_meta("d1") or {}).get("devices", {})

    # ---------------- 日志 ----------------
    def log(self, msg: str) -> None:
        print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

    def _jsonl(self, obj: dict) -> None:
        self._log_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._log_fh.flush()

    # ---------------- 模型 ----------------
    def try_load_model(self) -> bool:
        if not (fmlp.out_dir(self.session) / "train_meta.json").exists():
            return False
        try:
            m = fmlp.LiveFMLP(self.session)
        except Exception as e:
            self.log(f"模型加载失败：{e}")
            self.model_state = "failed"
            return False
        with self.lock:
            self.model, self.model_state = m, "ready"
        self.calib["calibrated_at"] = m.meta.get("calibrated_at") or self.calib.get("calibrated_at")
        self.log(f"模型已加载：{fmlp.out_dir(self.session)}（训练于 {m.meta.get('trained_at')}，"
                 f"标定轮 {m.meta.get('calib_round')}，自检拟合 {m.meta.get('calib_fit_acc', 0):.0%}）")
        return True

    def _calib_watch(self) -> None:
        """标定阶段：每 2 秒读一次采集记录；整轮采完后自动训练。"""
        last_n = -1
        while not self._stop.is_set() and self.model_state == "calibrating":
            try:
                self.calib.update(calib_progress(self.session, self.calib_round, self.grid.num_classes))
            except Exception as e:
                self.log(f"读标定进度出错：{e}")
            if self.calib["n_done"] != last_n:
                last_n = self.calib["n_done"]
                self.log(f"标定进度 {last_n} / {self.grid.num_classes}")
            if self.calib["complete"] and self.auto_train:
                self._run_training()
                return
            self._stop.wait(2.0)

    def _run_training(self) -> None:
        self.model_state = "training"
        self.log("标定轮已采完，开始解析 + 训练（几分钟）……")
        cmd = [sys.executable, "-X", "utf8", "-m", "backend.live.train",
               "--session", self.session, "--round", str(self.calib_round)]
        p = subprocess.run(cmd, cwd=str(config.PROJECT_ROOT))
        if p.returncode == 0 and self.try_load_model():
            self.log("训练完成，进入 ready")
        else:
            self.model_state = "failed"
            self.log(f"训练失败（返回码 {p.returncode}），看上面的输出。修好后可手动运行："
                     f"python -X utf8 -m backend.live.train --session {self.session} --round {self.calib_round}，"
                     f"再重启后端")

    # ---------------- 状态 ----------------
    def status(self) -> dict:
        now = time.time()
        links = self.buf.health(now)
        n_on = sum(v["online"] for v in links.values())
        ms, progress, msg = self.model_state, None, None
        if ms == "calibrating":
            state = "calibrating"
            n = self.calib["n_done"]
            progress = round(n / self.grid.num_classes, 3)
            msg = f"Calibrating: {n} / {self.grid.num_classes} points"
        elif ms == "training":
            state, msg = "training", "Training the model on today's calibration round"
        elif n_on == 0:
            state, msg = "offline", "No CSI from either Pi"
        elif ms == "failed":
            state, msg = "offline", "Model not available (training failed); see backend console"
        elif n_on < len(LINKS):
            down = [l for l, v in links.items() if not v["online"]]
            state, msg = "degraded", "Link down: " + ", ".join(down) + "; predictions paused"
        else:
            state = "ready"
        if ms in ("calibrating", "training") and n_on < len(LINKS):
            msg += f" (live data: {n_on} / 4 links online)"
        return {"type": "status", "contract_version": CONTRACT_VERSION, "source": self.source_kind,
                "session_id": self.session_id, "timestamp": iso(now), "state": state,
                "progress": progress, "calibrated_at": self.calib.get("calibrated_at"),
                "links": links, "message": msg}

    def meta(self) -> dict:
        g = self.grid
        cells = [{"label": l, "x": l % g.cols, "y": l // g.cols,
                  "x_m": round((l % g.cols) * g.spacing_m, 3), "y_m": round((l // g.cols) * g.spacing_m, 3)}
                 for l in range(g.num_classes)]
        devices = [{"id": k, "kind": "pi" if k.startswith("pi") else "ap",
                    "x_m": float(v["x"]), "y_m": float(v["y"]), "h_m": float(v["h"])}
                   for k, v in sorted(self.devices.items())]
        return {"contract_version": CONTRACT_VERSION, "source": self.source_kind, "session_id": self.session_id,
                "grid": {"cols": g.cols, "rows": g.rows, "spacing_m": g.spacing_m},
                "cells": cells, "devices": devices,
                "links": [{"id": l, "pi": l.split("_")[0], "ap": l.split("_")[1]} for l in LINKS],
                "window_frames": config.STAGE2_TIME_WINDOW, "frame_rate_hz": 10}

    # ---------------- 预测循环 ----------------
    def _predict_loop(self) -> None:
        last_totals = None
        last_report = 0.0
        while not self._stop.wait(STEP_S):
            now = time.time()
            if now - last_report >= 10:
                last_report = now
                h = self.buf.health(now)
                lp = self.latest
                self.log(f"[{self.status()['state']}] 帧率 " + "  ".join(f"{l} {v['fps']:.1f}" for l, v in h.items())
                         + (f"  | 最近预测 #{lp['seq']} label {lp['label']} ({lp['confidence']:.2f})" if lp else ""))
            if self.model_state != "ready" or self.model is None:
                continue
            w = self.buf.window(now=now)
            if w is None:
                if self.paused_reason != "window":
                    self.paused_reason = "window"
                    self.log("暂停出预测：有链路超过 2 秒没有数据，或帧还不够 16 帧")
                continue
            amp, phase, end, span, totals = w
            if last_totals is not None and any(totals[l] <= last_totals[l] for l in LINKS):
                continue                                     # 有链路没有新帧：不重复出同一个窗口
            last_totals = totals
            if self.paused_reason is not None:
                self.paused_reason = None
                self.log("恢复出预测")
            try:
                p = self.model.predict_proba_raw(amp, phase)[0]
            except Exception as e:
                self.log(f"推理出错：{e}")
                continue
            pred = self._prediction(p, end, span)
            with self.lock:
                self.latest = pred
            self._jsonl(pred)

    def _prediction(self, p: np.ndarray, end: float, span: float) -> dict:
        g = self.grid
        probs = [round(float(v), 4) for v in p]
        label = int(np.argmax(probs))
        xs = np.array([(l % g.cols) * g.spacing_m for l in range(g.num_classes)])
        ys = np.array([(l // g.cols) * g.spacing_m for l in range(g.num_classes)])
        pa = np.array(probs)
        t_end = iso(end)
        pred = {"type": "prediction", "contract_version": CONTRACT_VERSION, "source": self.source_kind,
                "session_id": self.session_id, "model_id": fmlp.MODEL_ID, "seq": self.seq,
                "timestamp": t_end,
                "window": {"start": iso(end - span), "end": t_end, "frames": config.STAGE2_TIME_WINDOW},
                "label": label, "x": label % g.cols, "y": label // g.cols,
                "x_m": round((label % g.cols) * g.spacing_m, 3), "y_m": round((label // g.cols) * g.spacing_m, 3),
                "confidence": probs[label], "probs": probs,
                "expected_m": {"x": round(float(pa @ xs), 3), "y": round(float(pa @ ys), 3)},
                "links_used": list(LINKS), "state": "ready"}
        self.seq += 1
        return pred

    # ---------------- 启停 ----------------
    def start(self) -> None:
        if not self.try_load_model() and self.model_state != "failed":
            self.model_state = "calibrating"
            self.log(f"{self.session} 还没有训练好的模型 → 等待标定轮 r{self.calib_round}"
                     f"（{'采完后自动训练' if self.auto_train else '不自动训练'}）")
            threading.Thread(target=self._calib_watch, name="calib", daemon=True).start()
        threading.Thread(target=self._predict_loop, name="predict", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._log_fh.close()


def make_handler(be: LiveBackend):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, obj: dict | None = None) -> None:
            body = b"" if obj is None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            if obj is not None:
                self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/api/meta":
                self._send(200, be.meta())
            elif path == "/api/status":
                self._send(200, be.status())
            elif path == "/api/predict/latest":
                with be.lock:
                    latest = be.latest
                st = be.status()["state"]
                if latest is None or st not in ("ready", "degraded"):
                    self._send(204)
                else:
                    self._send(200, latest)
            elif path == "/ws/predictions":
                self._send(501, {"error": "WebSocket is not implemented in this backend version; "
                                          "poll /api/status and /api/predict/latest instead"})
            else:
                self._send(404, {"error": f"unknown path {path}"})

        def log_message(self, fmt, *args):      # 客户端频繁轮询，不逐条打印
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="实时定位后端（F-MLP，HTTP 接口）")
    ap.add_argument("--session", required=True, help="当天的 session 名，例如 d14")
    ap.add_argument("--round", type=int, default=1, help="标定轮编号（默认 1）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-auto-train", action="store_true", help="标定采完后不自动训练")
    ap.add_argument("--replay", default=None, help="测试用：回放这个目录里的 pcap，不连 Pi")
    ap.add_argument("--replay-labels", default=None, help="回放哪些格点，例如 0,12")
    ap.add_argument("--replay-speed", type=float, default=1.0)
    args = ap.parse_args()

    kind = "replay" if args.replay else "live"
    be = LiveBackend(args.session, args.round, not args.no_auto_train, kind,
                     fmlp.out_dir(args.session))
    be.log(f"session={args.session}  source={kind}  session_id={be.session_id}")
    be.log(f"预测日志：{be.log_path}")

    if args.replay:
        from backend.live.sources import ReplaySource
        labels = [int(x) for x in args.replay_labels.split(",")] if args.replay_labels else None
        srcs = [ReplaySource(Path(args.replay), be.buf.add, be.log, labels=labels, speed=args.replay_speed)]
    else:
        from backend.live.sources import SshTcpdumpSource
        pi_ips = config.require_pi_ips()
        srcs = [SshTcpdumpSource(pi, pi_ips[pi], be.buf.add, be.log) for pi in config.PI_IDS]
    for s in srcs:
        s.start()
    be.start()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(be))
    httpd.daemon_threads = True
    be.log(f"HTTP 服务：http://{args.host}:{args.port}  （手机填这台电脑在局域网里的 IP，端口 {args.port}）")
    be.log("按 Ctrl+C 停止")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        be.log("正在停止……")
    finally:
        httpd.server_close()
        for s in srcs:
            s.stop()
        be.stop()


if __name__ == "__main__":
    main()
