"""通过 motorbridge 官方状态快照 ABI 读取反馈；不创建 socket 或解析 CAN。

ABI 来源：motor_abi/src/{lib,state_ffi}.rs 的 MotorStateMetadata 和
motor_handle_get_state_snapshot。旧 SDK 缺少该符号时由上层使用真实寄存器查询。
"""
from collections import deque
from dataclasses import dataclass
import ctypes
import math
import time

import numpy as np
from motorbridge.abi import CState
from motorbridge.errors import CallError


class FeedbackUnavailable(RuntimeError):
    pass


@dataclass
class FeedbackSnapshot:
    positions: np.ndarray
    times: np.ndarray
    sequences: np.ndarray


class StateMetadata(ctypes.Structure):
    _fields_ = [('feedback_age_ns', ctypes.c_uint64),
                ('feedback_sequence', ctypes.c_uint64)]


class MotorbridgeFeedback:
    @staticmethod
    def supported(motors) -> bool:
        return all(hasattr(getattr(getattr(m, '_abi', None), 'lib', None),
                           'motor_handle_get_state_snapshot') for m in motors)

    def __init__(self, robot, names) -> None:
        self.robot, self.names = robot, list(names)
        self.motors = [robot._motor_map[name] for name in names]
        if not self.supported(self.motors):
            raise ValueError('motorbridge 原生库不支持状态快照元数据')
        self.readers = []
        for motor in self.motors:
            getter = motor._abi.lib.motor_handle_get_state_snapshot
            getter.argtypes = [ctypes.c_void_p, ctypes.POINTER(CState),
                               ctypes.POINTER(StateMetadata)]
            getter.restype = ctypes.c_int32
            self.readers.append(getter)
        n = len(self.motors)
        self.q = np.zeros(n)
        self.times = np.full(n, -np.inf)
        self.sequences = np.zeros(n, dtype=np.uint64)
        self.intervals = [deque(maxlen=2048) for _ in self.motors]

    def snapshot(self, max_age: float) -> FeedbackSnapshot:
        for ctrl in self.robot._ctrl_map.values():
            ctrl.poll_feedback_once()  # 仍由 SDK 接收、解析、维护状态。
        for i, (motor, getter) in enumerate(zip(self.motors, self.readers)):
            state, metadata = CState(), StateMetadata()
            before = time.perf_counter()
            rc = getter(motor._require_open(), ctypes.byref(state), ctypes.byref(metadata))
            if rc != 0:
                raise CallError(f'{self.names[i]} motorbridge 状态快照读取失败: rc={rc}')
            if not state.has_value or not metadata.feedback_sequence:
                raise FeedbackUnavailable(f'{self.names[i]} 未收到 SDK 状态反馈')
            if state.status_code:
                raise CallError(f'{self.names[i]} SDK 电机故障标志 0x{state.status_code:02x}')
            if not math.isfinite(state.pos):
                raise CallError(f'{self.names[i]} SDK 位置反馈无效')
            age = metadata.feedback_age_ns/1e9
            if age > max_age:
                raise FeedbackUnavailable(f'{self.names[i]} SDK 反馈过期: {1000*age:.2f}ms')
            sequence = metadata.feedback_sequence
            if sequence < self.sequences[i]:
                raise CallError(f'{self.names[i]} SDK 反馈序号倒退')
            if sequence == self.sequences[i]:
                if float(state.pos) != self.q[i]:
                    raise CallError(f'{self.names[i]} 同一 SDK 序号的位置发生变化')
                continue  # 重复读取不刷新采样时间。
            received_at = before-age
            if received_at <= self.times[i]:
                raise CallError(f'{self.names[i]} SDK 反馈时间倒退')
            if self.sequences[i]:
                # SDK 序号计数真实报文；主机一次跳过多帧时不能误认为反馈变慢。
                count = sequence-int(self.sequences[i])
                self.intervals[i].append((received_at-self.times[i])/count)
            self.q[i], self.times[i], self.sequences[i] = state.pos, received_at, sequence
        if np.any(time.perf_counter()-self.times > max_age):
            raise FeedbackUnavailable('SDK 缓存反馈过期，未将重复读取视为新帧')
        return FeedbackSnapshot(self.q.copy(), self.times.copy(), self.sequences.copy())

    def reset_intervals(self) -> None:
        for gaps in self.intervals:
            gaps.clear()

    def rate_capacity(self) -> float:
        if any(len(gaps) < 3 for gaps in self.intervals):
            raise FeedbackUnavailable('SDK 新反馈样本不足，无法确定实际更新频率')
        slowest = min(1./float(np.mean(gaps)) for gaps in self.intervals)
        return float(round(slowest))

    def max_observed_period(self) -> float:
        if any(len(gaps) < 3 for gaps in self.intervals):
            raise FeedbackUnavailable('SDK 新反馈样本不足')
        return max(max(gaps) for gaps in self.intervals)

    def describe(self) -> str:
        slowest = min(1./float(np.mean(gaps)) for gaps in self.intervals)
        worst = max(max(gaps) for gaps in self.intervals)
        return f'feedback_slowest={slowest:.1f}Hz observed_period_max={1000*worst:.2f}ms (motorbridge snapshot)'
