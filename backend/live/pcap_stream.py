"""流式 pcap 解析：字节流可以分任意多块送进来，每凑齐一个完整的包就解出一帧。

判据与离线解析 backend/data/load_real_stage2._read_pcap_split 完全相同：
    - pcap 文件头的 magic 决定字节序与时间戳精度（微秒 / 纳秒）
    - 包长 < CSI_OFF + CSI_LEN 的跳过
    - data[48:52] 的 4 字节判别值决定 AP（apX / apY），其他设备的帧丢弃
    - I/Q：int16 小端交错，复数 = v[:,1] + 1j * v[:,0]
测试 tests/test_live.py 会把整份 pcap 分块送进来，与离线解析逐帧比对。
"""
from __future__ import annotations

import struct

import numpy as np

from backend.data.load_real import CSI_LEN, CSI_OFF, RAW_SUB
from backend.data.load_real_stage2 import _ap_discriminators

_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}


class PcapStreamParser:
    """feed(bytes) -> [(pcap 时间戳 s, ap_id, csi complex64[64]), ...]"""

    def __init__(self, ap_discriminators: dict[bytes, str] | None = None) -> None:
        self._buf = bytearray()
        self._endian: str | None = None
        self._scale = 1e-6
        self._ap_discriminators = ap_discriminators or _ap_discriminators()
        self.n_packets = 0          # 读到的包（含其他设备、过短的）
        self.n_frames = 0           # 输出的 apX / apY 帧

    def feed(self, chunk: bytes) -> list[tuple[float, str, np.ndarray]]:
        self._buf.extend(chunk)
        out: list[tuple[float, str, np.ndarray]] = []
        if self._endian is None:
            if len(self._buf) < 24:
                return out
            magic = bytes(self._buf[:4])
            if magic not in _MAGIC:
                raise ValueError(f"不是 pcap 流（magic={magic.hex()}）")
            self._endian, self._scale = _MAGIC[magic]
            del self._buf[:24]
        pos = 0
        n = len(self._buf)
        while n - pos >= 16:
            ts_sec, ts_frac, incl, _ = struct.unpack_from(f"{self._endian}IIII", self._buf, pos)
            if incl > 1_000_000:
                raise ValueError(f"pcap 包长异常：{incl}")
            if n - pos - 16 < incl:
                break
            start = pos + 16
            pos = start + incl
            self.n_packets += 1
            if incl < CSI_OFF + CSI_LEN:
                continue
            ap_id = self._ap_discriminators.get(bytes(self._buf[start + 48:start + 52]))
            if ap_id is None:
                continue
            v = np.frombuffer(bytes(self._buf[start + CSI_OFF:start + CSI_OFF + CSI_LEN]), dtype="<i2")
            v = v.astype(np.float32).reshape(RAW_SUB, 2)
            csi = (v[:, 1] + 1j * v[:, 0]).astype(np.complex64)
            out.append((ts_sec + ts_frac * self._scale, ap_id, csi))
            self.n_frames += 1
        del self._buf[:pos]
        return out
