"""Timestamped, receive-only RobStride private-protocol feedback on SocketCAN.

Commands still go through motorbridge. A separate CAN socket gets its own copy
of frames, so this receiver never consumes the SDK's reply queue. Status (2)
and active-report (24) have the same payload. Position decoding follows
https://github.com/tianrking/motorbridge/blob/main/motor_vendors/robstride/src/protocol.rs
(`decode_status_frame`: big-endian u16 / 0x7fff, range +/-4*pi).

Linux kernel receive timestamps are required: time spent in a queued packet
must not be mistaken for newly received feedback. No CAN frames are sent here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import socket
import struct
import threading
import time

import numpy as np

from .rebotarm import JointCfg

CAN_FRAME = struct.Struct('=IB3x8s')
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
SO_TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)  # Linux old-timespec ABI.
TIMESPEC = struct.Struct('@ll')


class FeedbackError(RuntimeError):
    pass


class FeedbackUnavailable(FeedbackError):
    """Missing/stale data; may be retried while motors are disabled at startup."""


@dataclass
class FeedbackSnapshot:
    positions: np.ndarray
    times: np.ndarray
    sequences: np.ndarray


class RSFeedbackCache:
    """A fresh packet, even with unchanged position, increments its sequence."""
    def __init__(self, joints: list[JointCfg]) -> None:
        if (not joints or any(j.vendor != 'robstride' for j in joints)
                or len({j.motor_id for j in joints}) != len(joints)
                or any(not 1 <= j.motor_id <= 255 or not 0 <= j.feedback_id <= 255 for j in joints)
                or any(j.model not in ('rs-00', 'rs-06') for j in joints)):
            raise ValueError('实时反馈当前仅支持独立 motor_id 的 RS00/RS06 关节')
        self.joints = joints
        self.index = {(j.motor_id, j.feedback_id): i for i, j in enumerate(joints)}
        n = len(joints)
        self.q = np.zeros(n)
        self.times = np.full(n, -np.inf)
        self.sequences = np.zeros(n, dtype=np.int64)
        self.intervals = [deque(maxlen=2048) for _ in joints]
        self.error: str | None = None
        self.lock = threading.Lock()

    def ingest(self, can_id: int, payload: bytes, received_at: float) -> bool:
        if (not can_id & CAN_EFF_FLAG or can_id & (CAN_ERR_FLAG | CAN_RTR_FLAG)
                or len(payload) != 8 or not math.isfinite(received_at)):
            return False
        kind = (can_id >> 24) & 0x1f
        if kind not in (2, 24):
            return False
        if kind == 2 and payload[:3] == b'\x00\xc4\x56':
            return False
        i = self.index.get(((can_id >> 8) & 0xff, can_id & 0xff))
        if i is None:
            return False
        faults = (can_id >> 16) & 0x3f
        position = (int.from_bytes(payload[:2], 'big')/0x7fff-1.)*(4.*math.pi)
        with self.lock:
            if faults:
                self.error = f'{self.joints[i].name} 状态故障标志 0x{faults:02x}'
            if received_at <= self.times[i]:
                return False  # Do not refresh age using out-of-order packets.
            if self.sequences[i]:
                self.intervals[i].append(received_at-self.times[i])
            self.q[i], self.times[i] = position, received_at
            self.sequences[i] += 1
        return True

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message

    def snapshot(self, now: float | None = None, max_age: float = .015) -> FeedbackSnapshot:
        with self.lock:
            if now is None:
                now = time.perf_counter()
            if not math.isfinite(now) or not math.isfinite(max_age) or max_age <= 0.:
                raise FeedbackError('反馈年龄检查参数无效')
            if self.error:
                raise FeedbackError(self.error)
            ages = now-self.times
            bad = np.flatnonzero((ages > max_age) | (ages < -.001))
            if len(bad):
                names = ', '.join(self.joints[i].name for i in bad)
                details = ', '.join(f'{self.joints[index].name}:{1000*ages[index]:.2f}ms' for index in bad)
                raise FeedbackUnavailable(f'实时反馈缺失/过期: {names}; ages={details}; '
                                          f'max_age={max_age*1000:.1f}ms')
            return FeedbackSnapshot(self.q.copy(), self.times.copy(), self.sequences.copy())

    def reset_intervals(self) -> None:
        with self.lock:
            for intervals in self.intervals:
                intervals.clear()

    def timing(self, rate: float, *, require_rate: bool = True) -> str:
        with self.lock:
            intervals = [list(gaps) for gaps in self.intervals]
        if any(len(gaps) < 3 for gaps in intervals):
            raise FeedbackUnavailable('主动反馈样本不足，无法验证实际反馈周期；未使能电机')
        worst = max(max(gaps) for gaps in intervals)
        slowest_hz = min(1./float(np.mean(gaps)) for gaps in intervals)
        if require_rate and (worst > 1.5/rate or slowest_hz < .9*rate):
            raise FeedbackUnavailable(
                f'反馈速率不足: slowest={slowest_hz:.1f}Hz max_gap={worst*1000:.2f}ms '
                f'目标={rate:g}Hz；未使能电机。可降低 --rate 或使用 --feedback query')
        return f'feedback_slowest={slowest_hz:.1f}Hz max_gap={worst*1000:.2f}ms'

    def rate_capacity(self) -> float:
        with self.lock:
            if any(len(gaps) < 3 for gaps in self.intervals):
                raise FeedbackUnavailable('SocketCAN 新反馈样本不足')
            return min(1./float(np.mean(gaps)) for gaps in self.intervals)

    def max_observed_period(self) -> float:
        with self.lock:
            if any(len(gaps) < 3 for gaps in self.intervals):
                raise FeedbackUnavailable('SocketCAN 新反馈样本不足')
            return max(max(gaps) for gaps in self.intervals)

    def describe(self) -> str:
        return self.timing(1., require_rate=False)+' (SocketCAN kernel timestamps)'


class SocketCANFeedback(RSFeedbackCache):
    def __init__(self, channel: str, joints: list[JointCfg]) -> None:
        super().__init__(joints)
        if not hasattr(socket, 'AF_CAN') or TIMESPEC.size != 16:
            raise FeedbackError('高速反馈需要 64 位 Linux SocketCAN；可使用 --feedback query')
        self.channel = channel
        self.sock: socket.socket | None = None
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None
        self.clock_sample: tuple[float, int, float] | None = None

    @staticmethod
    def _sample_clock() -> tuple[float, int, float]:
        before = time.perf_counter()
        wall = time.time_ns()
        return before, wall, time.perf_counter()

    def start(self) -> None:
        # Establish the clock relationship before packets can enter the socket.
        self.clock_sample = self._sample_clock()
        sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            # Match frame type, device and host, but allow all mode/fault bits.
            filters = b''.join(struct.pack('=II', CAN_EFF_FLAG | (kind << 24)
                                          | (j.motor_id << 8) | j.feedback_id,
                                          CAN_EFF_FLAG | CAN_RTR_FLAG
                                          | 0x1f00ffff)
                               for j in self.joints for kind in (2, 24))
            sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
            sock.settimeout(.02)
            sock.bind((self.channel,))
        except BaseException:
            sock.close()
            raise
        self.sock = sock
        self.thread = threading.Thread(target=self._receive, daemon=True, name='rs-feedback')
        self.thread.start()

    def _receive(self) -> None:
        try:
            if self.clock_sample is None:
                self.clock_sample = self._sample_clock()
            while not self.stopped.is_set():
                try:
                    frame, ancillary, flags, _ = self.sock.recvmsg(
                        CAN_FRAME.size, socket.CMSG_SPACE(TIMESPEC.size))
                except socket.timeout:
                    continue
                if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                    raise FeedbackError('SocketCAN 反馈或时间戳截断')
                if len(frame) != CAN_FRAME.size:
                    raise FeedbackError('SocketCAN 反馈帧长度错误')
                received_ns = None
                for level, kind, data in ancillary:
                    if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS:
                        seconds, nanos = TIMESPEC.unpack(data[:TIMESPEC.size])
                        received_ns = seconds*1_000_000_000+nanos
                if received_ns is None:
                    raise FeedbackError('缺少内核接收时间戳，不能验证反馈新鲜度')
                before, wall, after = self._sample_clock()
                previous_before, previous_wall, previous_after = self.clock_sample
                elapsed = after-previous_after
                uncertainty = (after-before)+(previous_after-previous_before)
                # Compare both clocks: a backward wall-clock step can otherwise
                # cancel a queued packet's age and falsely make it look fresh.
                # Allow sampling preemption and gradual clock slewing (1 ms/s).
                if abs((wall-previous_wall)/1e9-elapsed) > .001+uncertainty+.001*abs(elapsed):
                    raise FeedbackError('系统时钟跳变，反馈时间戳无效')
                self.clock_sample = before, wall, after
                age = (wall-received_ns)/1e9
                if age < -.001:
                    raise FeedbackError('系统时钟跳变，反馈时间戳无效')
                received_at = before-max(0., age)
                can_id, dlc, payload = CAN_FRAME.unpack(frame)
                if dlc == 8:
                    self.ingest(can_id, payload, received_at)
        except Exception as exc:
            if not self.stopped.is_set():
                self.fail(f'SocketCAN 接收失败: {exc}')

    def close(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=.1)
        if self.sock is not None:
            self.sock.close()
