"""数据源：每台 Pi 一个线程，把解析出的帧交给回调 on_frame(pi_id, ap_id, pcap_ts, csi)。

SshTcpdumpSource（正式）
    笔记本通过 SSH 在 Pi 上运行 tcpdump，把 pcap 字节流写到标准输出（-w -），实时传回。
    Pi 上不需要新脚本，只要开机后照常跑过 csi_setup.sh。断开后每 3 秒自动重连。
    与采集脚本 csi_capture.sh 用的是同一条过滤规则（dst port 5500）。两边可以同时运行：
    每个 tcpdump 各自拿到一份副本，互不影响（标定轮采集时后端也在取数，用来显示链路状态）。

ReplaySource（测试）
    回放已采的 pcap（例如 data/smoke/phase2_r1），按原始时间间隔送出，循环播放。
    只用于在没有 Pi 的环境里检查整条流程；回放时接口里的 source 是 "replay"。
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from backend.live.pcap_stream import PcapStreamParser

OnFrame = Callable[[str, str, float, np.ndarray], None]
Log = Callable[[str], None]

REMOTE_CMD = "sudo tcpdump -i wlan0 dst port 5500 -U -w - 2>/dev/null"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]


class SshTcpdumpSource(threading.Thread):
    def __init__(self, pi_id: str, host: str, on_frame: OnFrame, log: Log, user: str = "pi",
                 ssh_exe: str = "ssh", retry_s: float = 3.0):
        super().__init__(name=f"ssh-{pi_id}", daemon=True)
        self.pi_id, self.host, self.user = pi_id, host, user
        self.on_frame, self.log = on_frame, log
        self.ssh_exe, self.retry_s = ssh_exe, retry_s
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self.connects = 0

    def stop(self) -> None:
        self._stop.set()
        p = self._proc
        if p is not None and p.poll() is None:
            p.kill()

    def run(self) -> None:
        cmd = [self.ssh_exe, *SSH_OPTS, f"{self.user}@{self.host}", REMOTE_CMD]
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE, bufsize=0)
            except OSError as e:
                self.log(f"[{self.pi_id}] 无法启动 ssh：{e}")
                self._stop.wait(self.retry_s)
                continue
            self.connects += 1
            self.log(f"[{self.pi_id}] 已连接 {self.host}（第 {self.connects} 次），开始取数")
            parser = PcapStreamParser()
            try:
                while not self._stop.is_set():
                    chunk = self._proc.stdout.read(65536)
                    if not chunk:
                        break
                    for ts, ap, csi in parser.feed(chunk):
                        self.on_frame(self.pi_id, ap, ts, csi)
            except Exception as e:                       # 解析异常也不让线程死掉
                self.log(f"[{self.pi_id}] 数据流出错：{e}")
            if self._proc.poll() is None:
                self._proc.kill()
            err = b""
            try:
                err = self._proc.stderr.read() or b""
            except Exception:
                pass
            if not self._stop.is_set():
                tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
                self.log(f"[{self.pi_id}] 连接断开（已收 {parser.n_frames} 帧）{tail[0]}；"
                         f"{self.retry_s:.0f} 秒后重连")
                self._stop.wait(self.retry_s)


_FILE = re.compile(r"label(\d+)_pi(\d)\.pcap$")
_PI = {"1": "piA", "2": "piB"}


class ReplaySource(threading.Thread):
    """两台 Pi 的回放放在同一个线程里，保证两边同步。labels=None 表示目录里全部格点，按编号顺序。"""

    def __init__(self, round_dir: Path, on_frame: OnFrame, log: Log, labels: list[int] | None = None,
                 speed: float = 1.0, loop: bool = True):
        super().__init__(name="replay", daemon=True)
        self.round_dir, self.on_frame, self.log = Path(round_dir), on_frame, log
        self.speed, self.loop = speed, loop
        self._stop = threading.Event()
        files: dict[int, dict[str, Path]] = {}
        for f in sorted(self.round_dir.glob("label*_pi*.pcap")):
            m = _FILE.search(f.name)
            if m and m.group(2) in _PI:
                files.setdefault(int(m.group(1)), {})[_PI[m.group(2)]] = f
        if labels is not None:
            files = {k: v for k, v in files.items() if k in labels}
        if not files:
            raise FileNotFoundError(f"{self.round_dir} 下没有可回放的 label*_pi*.pcap")
        self.timeline = self._build(files)
        self.current_label: int | None = None
        self.log(f"[回放] {self.round_dir}：格点 {sorted(files)}，共 {len(self.timeline)} 帧")

    @staticmethod
    def _build(files: dict[int, dict[str, Path]]):
        """[(相对时间 s, 格点, pi, ap, pcap_ts, csi)]，各格点首尾相接。"""
        out, offset = [], 0.0
        for lab in sorted(files):
            frames = []
            for pi, f in files[lab].items():
                p = PcapStreamParser()
                frames += [(ts, pi, ap, csi) for ts, ap, csi in p.feed(f.read_bytes())]
            if not frames:
                continue
            t0 = min(fr[0] for fr in frames)
            dur = max(fr[0] for fr in frames) - t0
            for ts, pi, ap, csi in frames:
                out.append((offset + ts - t0, lab, pi, ap, ts, csi))
            offset += dur + 0.1
        out.sort(key=lambda r: r[0])
        return out

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            start = time.monotonic()
            for rel, lab, pi, ap, ts, csi in self.timeline:
                wait = rel / self.speed - (time.monotonic() - start)
                if wait > 0 and self._stop.wait(wait):
                    return
                if lab != self.current_label:
                    self.current_label = lab
                    self.log(f"[回放] 现在播放格点 {lab}")
                self.on_frame(pi, ap, ts, csi)
            if not self.loop:
                return
