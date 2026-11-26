"""四条链路的帧缓冲与链路状态。

窗口的取法：每条链路各取**最近的 16 帧**，按链路顺序拼成 [T=16, L=4, 64]。
离线数据（load_real_stage2）是"同一格点内四条链路按包序号截断对齐，再不重叠地每 16 帧切一窗"。
人站着不动时，一个窗口约 1.6 秒内信道是准静态的，两种取法看到的是同一个信道，所以等价。
人在走动时，四条链路的最近 16 帧并不严格同时，这是实时演示的已知近似（离线数据里没有走动）。
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from backend import config

LINKS = config.all_link_ids()          # ["piA_apX", "piA_apY", "piB_apX", "piB_apY"]
ONLINE_S = 2.0                         # 约定 v0.1：最近 2 秒内收到过数据才算在线
FPS_WINDOW_S = 5.0


class LinkBuffers:
    def __init__(self, keep: int = 64):
        self._lock = threading.Lock()
        self._frames = {l: deque(maxlen=keep) for l in LINKS}     # (到达时间, pcap 时间戳, csi)
        self._arrivals = {l: deque() for l in LINKS}               # 最近 5 秒的到达时间（算帧率）
        self.total = {l: 0 for l in LINKS}

    def add(self, pi_id: str, ap_id: str, pcap_ts: float, csi: np.ndarray, now: float | None = None) -> None:
        link = f"{pi_id}_{ap_id}"
        if link not in self._frames:
            return
        now = time.time() if now is None else now
        with self._lock:
            self._frames[link].append((now, pcap_ts, csi))
            arr = self._arrivals[link]
            arr.append(now)
            while arr and arr[0] < now - FPS_WINDOW_S:
                arr.popleft()
            self.total[link] += 1

    def health(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        out = {}
        with self._lock:
            for l in LINKS:
                fr, arr = self._frames[l], self._arrivals[l]
                while arr and arr[0] < now - FPS_WINDOW_S:
                    arr.popleft()
                age = (now - fr[-1][0]) if fr else None
                out[l] = {"online": age is not None and age <= ONLINE_S,
                          "fps": round(len(arr) / FPS_WINDOW_S, 1),
                          "last_frame_age_s": round(age, 2) if age is not None else 999.0}
        return out

    def window(self, T: int = config.STAGE2_TIME_WINDOW, now: float | None = None):
        """四条链路都在线且各有 ≥ T 帧时，返回 (amp[T,L,64], phase[T,L,64], 窗口结束时刻, 窗口时长 s, 各链路总帧数)；
        否则返回 None。窗口结束时刻 = 四条链路中最新一帧的到达时间（笔记本时钟）；
        窗口时长 = 各链路 16 帧的 pcap 时间跨度的最大值（同一台 Pi 的时钟内相减，与对时无关）。"""
        now = time.time() if now is None else now
        with self._lock:
            snap = {}
            for l in LINKS:
                fr = self._frames[l]
                if len(fr) < T or now - fr[-1][0] > ONLINE_S:
                    return None
                snap[l] = list(fr)[-T:]
            totals = dict(self.total)
        csi = np.stack([np.stack([f[2] for f in snap[l]]) for l in LINKS], axis=1)   # [T,L,64]
        end = max(snap[l][-1][0] for l in LINKS)
        span = max(snap[l][-1][1] - snap[l][0][1] for l in LINKS)
        span = float(min(max(span, 0.0), 10.0))
        return (np.abs(csi).astype(np.float32), np.angle(csi).astype(np.float32), end, span, totals)
