"""RS 笛卡尔阻抗：平滑关节轨迹到位 → 末端阻抗 → Ctrl+C 平滑归零保持。

通信及 MIT 发送复用 actuator 的分组接口，IK 复用 kinematics，移动和归零
复用 trajectory.JointMotion。与 demo9/10 共用重力接口、最小 jerk 时间剖面。
运行故障优先进入受控保持；无可信反馈时保留电机端最后指令，等待人工接管，
不自动失能或重新使能。仅用户明确退出才失能。软件不能保证失联时的位置保持。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import queue
import select
import signal
import sys
import threading
import time
from typing import Callable

import numpy as np
import pinocchio as pin

from reBotArm_control_py.actuator import RebotArm, load_cfg, load_gravity_compensation_config
from reBotArm_control_py.actuator.motorbridge_feedback import MotorbridgeFeedback, FeedbackUnavailable
from reBotArm_control_py.actuator.rs_feedback import (
    SocketCANFeedback, FeedbackUnavailable as SocketFeedbackUnavailable,
)
from reBotArm_control_py.kinematics import (
    get_end_effector_frame, load_robot_model, solve_ik, pos_rot_to_se3, IKSolverParams,
)
from reBotArm_control_py.dynamics import compute_generalized_gravity, compute_nle
from reBotArm_control_py.trajectory import JointMotion, minimum_jerk_profile

# 按 arm 组顺序；均为电机角度。当前 RS 的电机与本仓库 URDF 正方向一致。
# gripper_end 的 +X 是夹爪前向；joint4=-20° 抵消原目标约 20° 的上仰。
INITIAL_JOINTS_DEG = [0.0, 40.0, 60.0, -20.0, 0.0, 0.0]
HOME_JOINTS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# 关节弹簧 N*m/rad、阻尼 N*m*s/rad；持续由电机 MIT 内环执行。
# Demo11 独立参数，不修改 demo3/8 的通用定位增益；YAML profile 可覆盖。
MIT_MOVE_KP = np.array([30., 70., 70., 15., 15., 10.])
MIT_MOVE_KD = np.array([3., 6., 6., 1., 1., .3])
MIT_HOLD_KP = np.array([7.5, 15., 15., 4., 4., 2.5])
MIT_HOLD_KD = MIT_MOVE_KD.copy()
TORQUE_LIMIT = np.array([12., 18., 18., 3., 3., 3.])  # 主机估计总力矩限幅 N*m
TORQUE_RATE_LIMIT = np.array([40., 60., 60., 10., 10., 10.])  # N*m/s
TORQUE_EPSILON = 1e-8  # 仅容纳浮点舍入误差，不放宽物理力矩限幅。
CONTROL_RATE = 200.0
LEGACY_CONTROL_RATE = 200.0
CARTESIAN_STIFFNESS = np.array([100., 100., 100., 3., 3., 3.])
CARTESIAN_DAMPING = np.array([12., 12., 12., .15, .4, .4])
GRIPPER_REFRESH_RATE = 50.
QUERY_CONTROL_RATE = 15.0  # 兼容旧 --feedback query 的默认期望频率。
PREFLIGHT_SAMPLES = 12
PREFLIGHT_WARMUP = 2
CONTROL_PHASES = ('MOVE', 'SETTLE', 'BLEND', 'IMPEDANCE', 'HOMING', 'PROTECTED_HOLD')
MIN_MOVE_DURATION = 6.0
MAX_MOVE_SPEED = .20  # rad/s，最小 jerk 的峰值速度
MAX_MOVE_ACCELERATION = .08  # rad/s²
START_HOLD_DURATION = .6  # 使能后先刷新当前位置保持，再开始轨迹。
SETTLE_TIME = .4
SETTLE_TIMEOUT = 8.0
ARRIVAL_TOLERANCE = np.deg2rad(3.)
BLEND_DURATION = 2.0
MAX_SAMPLE_DT = .15  # 超过此间隔不再使用该反馈继续运动，转为等待/恢复保持。
VELOCITY_FILTER_TIME_CONSTANT = .01
MAX_JOINT_SPEED = 3.0
POSITION_QUANTUM = 4.*np.pi/0x7fff
MIN_VELOCITY_WINDOW = .002  # 合并密集状态/主动报文，避免对亚毫秒量化跳变求导。
MAX_TRACKING_ERROR = np.deg2rad(20.)
MAX_POSITION_ERROR = .20  # m，超过则转为当前位置保护保持
MAX_ROTATION_ERROR = np.deg2rad(60.)  # rad，超过则转为当前位置保护保持


def resolve_rate(feedback_mode: str, rate: float | None = None, *, control_space: str = 'cartesian') -> float:
    if feedback_mode not in ('motorbridge', 'socketcan', 'query'):
        raise ValueError('feedback_mode 必须为 motorbridge、socketcan 或 query')
    default_rate = LEGACY_CONTROL_RATE if control_space == 'joint' else CONTROL_RATE
    value = (QUERY_CONTROL_RATE if feedback_mode == 'query' else default_rate) if rate is None else rate
    if not np.isfinite(value) or not 10. <= value <= 500.:
        raise ValueError('控制频率必须在 10..500 Hz 范围内')
    return float(value)


class ControlFault(RuntimeError):
    """控制或反馈异常，禁止继续正常轨迹。"""


class HoldRequest(ControlFault):
    """反馈仍有效，但应放弃当前轨迹/笛卡尔目标，进入受控位置保持。"""


class EnterKey:
    """非阻塞地检测换行；部分管道输入和 EOF 都不会阻塞控制监督线程。"""
    def __init__(self, stream) -> None:
        self.stream = stream
        self.fd: int | None = None
        self.was_blocking: bool | None = None
        self.available = False
        try:
            self.fd = stream.fileno()
            self.was_blocking = os.get_blocking(self.fd)
            os.set_blocking(self.fd, False)
            self.available = True
        except (OSError, ValueError, AttributeError):
            pass

    def poll(self) -> bool:
        if not self.available:
            return False
        try:
            if not select.select([self.fd], [], [], 0)[0]:
                return False
            chunk = os.read(self.fd, 4096)
            if not chunk:
                self.available = False
            return b'\n' in chunk
        except BlockingIOError:
            return False
        except (OSError, ValueError):
            self.available = False
            return False

    def close(self) -> None:
        if self.fd is not None and self.was_blocking is not None:
            try:
                if self.stream.fileno() == self.fd:
                    os.set_blocking(self.fd, self.was_blocking)
            except (OSError, ValueError):
                pass


class StopRequest:
    """SIGINT 只记录请求，不在 CAN 事务或一批电机指令中间抛异常。"""
    def __init__(self) -> None:
        self.count = 0

    def handle(self, signum, frame) -> None:
        self.count += 1

    def requested(self) -> bool:
        return self.count > 0


class LoopWatchdog:
    """监督成功发送的心跳；超时由上层锁存为等待人工接管，不自动失能。"""
    def __init__(self, rate: float, *, control_timeout: float | None = None) -> None:
        self.last_completed = time.perf_counter()
        self.hold_after = max(MAX_SAMPLE_DT, 2.5/rate) if control_timeout is None else control_timeout
        self.fail_after = max(.5, 8./rate)
        self.hold_requested = threading.Event()
        self.lock = threading.Lock()
        self.stage = '等待首帧'

    def set_stage(self, stage: str) -> None:
        with self.lock:
            self.stage = stage

    def complete(self) -> None:
        with self.lock:
            self.last_completed = time.perf_counter()
            self.hold_requested.clear()

    def poll(self) -> bool:
        with self.lock:
            age = time.perf_counter()-self.last_completed
            if age > self.fail_after:
                raise ControlFault(f'控制心跳超时 {age:.3f}s，阶段={self.stage}；'
                                   '持续无法更新控制指令，等待人工接管')
            if age > self.hold_after and not self.hold_requested.is_set():
                self.hold_requested.set()
                return True
        return False


class RuntimeFault:
    """故障锁存；代号变化使已暂停的旧 MIT 批次在恢复后停止发送。"""
    def __init__(self) -> None:
        self.pending = threading.Event()
        self.generation = 0
        self.reason = ''
        self.lock = threading.Lock()

    def request(self, reason: str) -> bool:
        with self.lock:
            if self.pending.is_set():
                return False
            self.generation += 1
            self.reason = reason
            self.pending.set()
            return True

    def complete(self, generation: int) -> None:
        with self.lock:
            if generation == self.generation:
                self.pending.clear()


class RecoverySamples:
    """故障后至少核对两批新反馈和速度，才允许捕获新的保持位置。"""
    def __init__(self) -> None:
        self.estimator = None

    def reset(self) -> None:
        self.estimator = None

    def ready(self, q: np.ndarray, now: float, sample_times: np.ndarray | None,
              *, reference: VelocityEstimator | None = None) -> bool:
        if self.estimator is None:
            candidate = VelocityEstimator()
            candidate.update(q, now, sample_times=sample_times)
            if reference is not None and reference.previous is not None:
                elapsed = candidate.sample_times-reference.sample_times
                uncertainty = POSITION_QUANTUM if sample_times is not None else 0.
                if (np.any(elapsed < 0.) or np.any(np.abs(q-reference.previous)
                        > MAX_JOINT_SPEED*elapsed+uncertainty+1e-12)
                        or np.any((elapsed == 0.) & (q != reference.previous))):
                    raise ControlFault('故障恢复首帧相对故障前反馈发生位置跳变或速度超过限制')
            self.estimator = candidate
            return False
        if sample_times is not None and not np.all(sample_times > self.estimator.sample_times):
            return False  # 同一缓存不能充当第二批可信反馈。
        self.estimator.update(q, now, sample_times=sample_times)
        return True


class BackgroundLog:
    """Bounded, best-effort logging; a blocked terminal never blocks post()."""
    def __init__(self, sink: Callable[[str], None] | None = None) -> None:
        self.messages: queue.Queue[str] = queue.Queue(maxsize=4)
        self.stopped = threading.Event()
        self.sink = sink if sink is not None else lambda msg: print(msg, flush=True)
        self.thread = threading.Thread(target=self._write, daemon=True, name='demo11-log')
        self.thread.start()

    def post(self, message: str) -> None:
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            # 丢弃状态日志，不等待终端；日志不能阻塞控制或触发失能。
            pass

    def _write(self) -> None:
        while not self.stopped.is_set() or not self.messages.empty():
            try:
                message = self.messages.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                self.sink(message)
            except Exception:
                self.stopped.set()
                return

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=.2)  # 仅在电机已失能后调用；终端阻塞时不无限等待。


@dataclass
class TimingReport:
    # columns: full arm feedback, all MIT writes, control math (seconds)
    costs: np.ndarray
    rate: float = CONTROL_RATE
    periods: np.ndarray | None = None

    @property
    def budget(self) -> float:
        return .8/self.rate

    @property
    def max_cycle(self) -> float:
        return float(np.sum(np.max(self.costs, axis=0)))

    @property
    def loop_capacity(self) -> float:
        if self.periods is None or not len(self.periods):
            return np.inf
        return float(round(1./np.mean(self.periods), 6))

    @property
    def max_period(self) -> float:
        return 0. if self.periods is None or not len(self.periods) else float(np.max(self.periods))

    def describe(self, selected_rate: float | None = None) -> str:
        medians = 1000*np.median(self.costs, axis=0)
        selected = self.select_rate() if selected_rate is None else selected_rate
        observed = (f'probe_hz={self.loop_capacity:.1f} probe_gap_max={1000*self.max_period:.2f}ms'
                    if self.periods is not None else 'probe_hz=n/a')
        return (f'[TIMING] median read={medians[0]:.2f}ms write={medians[1]:.2f}ms '
                f'math={medians[2]:.2f}ms; max_cycle={1000*self.max_cycle:.2f}ms '
                f'budget={1000*self.budget:.2f}ms @ requested {self.rate:g}Hz; '
                f'selected={selected:g}Hz ({1000/selected:.2f}ms); {observed}')

    def select_rate(self, feedback_capacity: float = np.inf) -> float:
        """留 20% 读写计算余量，正常通信较慢时降频而非失能。"""
        if not np.isfinite(self.max_cycle) or self.max_cycle <= 0:
            raise ControlFault('无效的通信周期测量')
        capacity = .8/self.max_cycle
        # 不让正常循环的名义周期超过既有 150 ms 保护窗口。
        # 极慢通信仍由单次读取时长/通信异常检查处理。
        return min(self.rate, max(8., float(np.floor(min(capacity, feedback_capacity, self.loop_capacity)))))

def minimum_jerk(q0: np.ndarray, q1: np.ndarray, elapsed: float,
                 duration: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration 必须为有限正数")
    blend, speed = minimum_jerk_profile(elapsed, duration)
    return q0 + (q1-q0)*blend, (q1-q0)*speed


def move_duration(q0: np.ndarray, q1: np.ndarray) -> float:
    # quintic s'(t) 的峰值为 1.875 / duration。
    return JointMotion.plan(q0, q1, max_speed=MAX_MOVE_SPEED,
                            min_duration=MIN_MOVE_DURATION,
                            max_acceleration=MAX_MOVE_ACCELERATION).duration


class VelocityEstimator:
    def __init__(self) -> None:
        self.previous: np.ndarray | None = None
        self.previous_time: float | None = None
        self.filtered = np.zeros(6)
        self.sample_times: np.ndarray | None = None
        self.anchor_q: np.ndarray | None = None
        self.anchor_times: np.ndarray | None = None

    def update(self, q: np.ndarray, now: float, *, sample_times: np.ndarray | None = None
               ) -> tuple[np.ndarray, float]:
        if q.shape != (6,) or not np.all(np.isfinite(q)) or not np.isfinite(now):
            raise ControlFault("速度估计输入无效")
        stamps = np.full(6, now) if sample_times is None else np.asarray(sample_times)
        if (stamps.shape != (6,) or not np.all(np.isfinite(stamps))
                or np.any(stamps > now+.001) or np.any(now-stamps > MAX_SAMPLE_DT)):
            raise ControlFault('关节反馈时间戳无效或过期')
        if self.previous is None:
            self.previous, self.previous_time = q.copy(), now
            self.sample_times = stamps.copy()
            self.anchor_q, self.anchor_times = q.copy(), stamps.copy()
            return self.filtered.copy(), 1. / CONTROL_RATE
        dt = now - self.previous_time
        if not 0. < dt <= MAX_SAMPLE_DT:
            raise ControlFault(f"采样周期异常: {dt:.4f}s")
        sample_dt = stamps-self.sample_times
        updated = sample_dt > 0.
        if (np.any(sample_dt < 0.) or np.any(sample_dt > MAX_SAMPLE_DT)
                or np.any((~updated) & (q != self.previous))):
            raise ControlFault('关节反馈时间倒退、丢帧过久或同一时间戳的位置改变')
        window = stamps-self.anchor_times
        displacement = q-self.anchor_q
        uncertainty = POSITION_QUANTUM if sample_times is not None else 0.
        # 两个量化位置的差最多有一个编码格误差。真实跳变仍立即拒绝，
        # 并非通过低通把超速隐藏掉；短窗口累积到至少 2 ms 才更新速度。
        if np.any(updated & (np.abs(displacement) > MAX_JOINT_SPEED*window+uncertainty+1e-12)):
            raise ControlFault("位置跳变或关节速度超过限制")
        ready = updated & (window >= MIN_VELOCITY_WINDOW-1e-12)
        raw = displacement[ready]/window[ready]
        alpha = 1. - np.exp(-window[ready]/VELOCITY_FILTER_TIME_CONSTANT)
        self.filtered[ready] += alpha * (raw-self.filtered[ready])
        self.anchor_q[ready], self.anchor_times[ready] = q[ready], stamps[ready]
        self.previous, self.previous_time = q.copy(), now
        self.sample_times = stamps.copy()
        return self.filtered.copy(), dt


class ControlRateMonitor:
    def __init__(self, rate: float, *, label: str = '实际控制') -> None:
        self.period = 1./rate
        self.label = label
        self.intervals: deque[float] = deque(maxlen=50)
        self.late_count = 0

    def observe(self, dt: float) -> None:
        self.intervals.append(dt)
        self.late_count = self.late_count+1 if dt > 1.5*self.period else 0
        if (self.late_count >= 3 or (len(self.intervals) == self.intervals.maxlen
                and np.mean(self.intervals) > 1.1*self.period)):
            raise HoldRequest(f'{self.label}频率持续不足：mean={1./np.mean(self.intervals):.1f}Hz，'
                              f'目标={1./self.period:g}Hz；暂停轨迹并保持当前位置')


@dataclass
class CartesianResult:
    tau: np.ndarray
    gravity: np.ndarray
    error: np.ndarray


class ImpedanceLaw:
    """只做模型和控制计算；构造/调用不连接 CAN，便于离线验证。"""
    def __init__(self, initial_deg: list[float] | np.ndarray, *, control_space: str = 'cartesian') -> None:
        if control_space not in ('cartesian', 'joint'):
            raise ValueError('control_space 必须为 cartesian 或 joint')
        self.control_space = control_space
        cfg = load_cfg()
        self.joint_names = cfg['groups']['arm']['joints']
        joints = {joint.name: joint for joint in cfg['joints']}
        if self.joint_names != [f'joint{i}' for i in range(1, 7)]:
            raise ValueError("本示例要求 arm 组为 joint1..joint6，顺序一致")
        if any(joints[name].vendor != 'robstride' for name in self.joint_names):
            raise ValueError("本例使用 RS mechPos 反馈读取，只支持当前 RobStride 配置")
        if any(joints[name].model not in ('rs-00', 'rs-06') for name in self.joint_names):
            raise ValueError('本示例仅核对了 RS00/RS06 的 SDK 编码范围')
        kp_limits = np.array([5000. if joints[name].model == 'rs-06' else 500.
                              for name in self.joint_names])
        kd_limits = np.array([100. if joints[name].model == 'rs-06' else 5.
                              for name in self.joint_names])
        self.model = load_robot_model()
        self.data = self.model.createData()
        self.frame_name = get_end_effector_frame()
        self.frame = self.model.getFrameId(self.frame_name)
        if self.frame >= self.model.nframes:
            raise ValueError(f"末端帧不存在: {self.frame_name}")
        q_indices, v_indices = [], []
        for name in self.joint_names:
            if not self.model.existJointName(name):
                raise ValueError(f"URDF 缺少关节: {name}")
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"{name} 必须是单自由度关节")
            q_indices.append(joint.idx_q)
            v_indices.append(joint.idx_v)
        self.q_indices = np.array(q_indices)
        self.v_indices = np.array(v_indices)
        self.lower = self.model.lowerPositionLimit[self.q_indices]
        self.upper = self.model.upperPositionLimit[self.q_indices]
        self.torque_limit = np.minimum(TORQUE_LIMIT, self.model.effortLimit[self.v_indices])
        if np.any(self.torque_limit <= 0) or not np.all(np.isfinite(self.torque_limit)):
            raise ValueError("无效的关节力矩限制")
        self.initial = np.deg2rad(np.asarray(initial_deg, dtype=float))
        self.validate_q(self.initial, strict=True)
        self.home = np.deg2rad(np.asarray(HOME_JOINTS_DEG, dtype=float))
        self.validate_q(self.home, strict=True)
        # Demo11 使用独立的重力补偿系数，避免影响 demo9/10。
        gc = load_gravity_compensation_config('cartesian_impedance')
        for name, fallback in (('cartesian_stiffness', CARTESIAN_STIFFNESS),
                               ('cartesian_damping', CARTESIAN_DAMPING)):
            value = np.asarray(gc.get(name, fallback), dtype=float)
            if value.shape != (6,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f'{name} 必须为 6 个正有限数值，顺序 x y z rx ry rz')
            setattr(self, name, value.copy())
        self.compensate_coriolis = gc.get('compensate_coriolis', True)
        if not isinstance(self.compensate_coriolis, bool):
            raise ValueError('compensate_coriolis 必须为布尔值')
        for name, fallback in (('move_kp', MIT_MOVE_KP), ('move_kd', MIT_MOVE_KD),
                               ('hold_kp', MIT_HOLD_KP), ('hold_kd', MIT_HOLD_KD)):
            value = np.asarray(gc.get('mit_'+name, fallback), dtype=float)
            if (value.shape != (6,) or not np.all(np.isfinite(value))
                    or np.any(value <= 0)):
                raise ValueError(f'mit_{name} 必须为 arm 顺序的 6 个正有限数值')
            limits = kp_limits if name.endswith('kp') else kd_limits
            if np.any(value > limits):
                raise ValueError(f'mit_{name} 超过当前 RS 型号的 SDK 编码范围 {limits}')
            setattr(self, name, value.copy())
        scale = np.asarray(gc.get('tau_scale', 1.), dtype=float)
        self.tau_scale = np.broadcast_to(scale, (6,)).copy()
        if not np.all(np.isfinite(self.tau_scale)) or np.any(self.tau_scale <= 0):
            raise ValueError("tau_scale 必须为正的有限标量或 6 个值")
        self.target = self.pose(self.initial)

    def set_target_pose(self, xyz_rpy: list[float]) -> None:
        """和 demo7/8 共用 IK：离线先求关节终点，再规划关节移动。RPY 为弧度。"""
        values = np.asarray(xyz_rpy, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError('笛卡尔目标必须为有限的 x y z roll pitch yaw')
        target = pos_rot_to_se3(values[:3], roll=values[3], pitch=values[4], yaw=values[5])
        result = solve_ik(self.model, self.data, self.frame, target,
                          self.model_q(self.initial), IKSolverParams(), controlled_joints=6)
        if not result.success:
            raise ValueError(f'目标位姿 IK 未收敛，未连接硬件: error={result.error:.3e}')
        initial = result.q[self.q_indices].copy()
        self.validate_q(initial, strict=True)
        pose = self.pose(initial)
        if (np.linalg.norm(pose.translation-target.translation) > .001
                or np.linalg.norm(pin.log3(target.rotation @ pose.rotation.T)) > .01):
            raise ValueError('IK 终点未满足目标位姿容差，未连接硬件')
        self.initial, self.target = initial, target.copy()

    def validate_q(self, q: np.ndarray, *, strict: bool = False) -> None:
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ControlFault("关节位置必须是 6 个有限数值")
        # 反馈有编码量化/零位误差，实测允许 2°，指定目标严格遵守 URDF 限位。
        margin = 0. if strict else np.deg2rad(2.)
        if np.any(q < self.lower-margin) or np.any(q > self.upper+margin):
            raise ControlFault(f"关节位置超出 URDF 限位: {np.rad2deg(q)} deg")

    def model_q(self, q: np.ndarray) -> np.ndarray:
        full = pin.neutral(self.model)
        full[self.q_indices] = q
        # 当前模型夹爪为末端的两个移动分支；无电机角->夹指位移映射，保持中性值。
        return full

    def pose(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, self.model_q(q))
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame].copy()

    def gravity(self, q: np.ndarray) -> np.ndarray:
        gravity = compute_generalized_gravity(self.model, self.model_q(q), self.data)
        return gravity[self.v_indices].copy() * self.tau_scale

    def geometry(self, q: np.ndarray) -> tuple[pin.SE3, np.ndarray]:
        full = self.model_q(q)
        pin.computeJointJacobians(self.model, self.data, full)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.frame].copy()
        # WORLD 的线速度参考点是世界原点；这里必须用末端原点的 LWA。
        jacobian = pin.getFrameJacobian(
            self.model, self.data, self.frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:, self.v_indices].copy()
        return pose, jacobian

    def cartesian(self, q: np.ndarray, qd: np.ndarray) -> CartesianResult:
        """基座坐标系、末端原点的弹簧阻尼；输出含重力/科氏补偿的关节力矩。"""
        self.validate_q(q)
        if qd.shape != (6,) or not np.all(np.isfinite(qd)):
            raise ControlFault('关节速度必须是 6 个有限数值')
        pose, jacobian = self.geometry(q)
        error = np.concatenate((self.target.translation-pose.translation,
                                pin.log3(self.target.rotation @ pose.rotation.T)))
        if (np.linalg.norm(error[:3]) > MAX_POSITION_ERROR
                or np.linalg.norm(error[3:]) > MAX_ROTATION_ERROR):
            raise HoldRequest(f'末端偏离固定目标过大: pos={np.linalg.norm(error[:3]):.3f}m '
                              f'rot={np.linalg.norm(error[3:]):.3f}rad')
        gravity = self.gravity(q)
        if self.control_space == 'joint':
            joint_error = self.tracking_error(q, self.initial)
            tau = gravity + self.hold_kp*joint_error - self.hold_kd*qd
            return CartesianResult(tau, gravity, error)
        spring = self.cartesian_stiffness*error
        spring[3:] = pin.Jlog3(self.target.rotation @ pose.rotation.T).T @ spring[3:]
        tau = gravity + jacobian.T @ (spring-self.cartesian_damping*(jacobian @ qd))
        if self.compensate_coriolis:
            velocity = np.zeros(self.model.nv)
            velocity[self.v_indices] = qd
            full = self.model_q(q)
            bias = compute_nle(self.model, full, velocity, self.data)[self.v_indices]
            nominal_gravity = compute_generalized_gravity(self.model, full, self.data)[self.v_indices]
            tau += bias-nominal_gravity
        return CartesianResult(tau, gravity, error)

    def tracking_error(self, q: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """关节弹簧在移动和阻抗保持阶段使用相同的偏差保护。"""
        error = ref-q
        worst = int(np.argmax(np.abs(error)))
        if abs(error[worst]) > MAX_TRACKING_ERROR:
            raise HoldRequest(f'关节轨迹跟踪误差超过 20°: {self.joint_names[worst]} '
                              f'error={np.rad2deg(error[worst]):.2f}deg')
        return error

    def joint_hold(self, q: np.ndarray, qd: np.ndarray, ref: np.ndarray,
                   ref_velocity: np.ndarray) -> np.ndarray:
        error = self.tracking_error(q, ref)
        return self.gravity(q) + self.move_kp*error + self.move_kd*(ref_velocity-qd)

    def limit_torque(self, tau: np.ndarray, previous: np.ndarray | None,
                     dt: float) -> tuple[np.ndarray, bool]:
        if not np.isfinite(dt) or not 0. < dt <= MAX_SAMPLE_DT:
            raise ControlFault("力矩限速的采样周期无效")
        if tau.shape != (6,) or not np.all(np.isfinite(tau)):
            raise ControlFault("力矩指令含无效数值")
        limited = np.clip(tau, -self.torque_limit, self.torque_limit)
        if previous is not None:
            if (previous.shape != (6,) or not np.all(np.isfinite(previous))
                    or np.any(np.abs(previous) > self.torque_limit+TORQUE_EPSILON)):
                raise ControlFault("前一力矩指令无效或已超限")
            previous = np.clip(previous, -self.torque_limit, self.torque_limit)
            step = TORQUE_RATE_LIMIT * dt
            limited = np.clip(limited, previous-step, previous+step)
        return limited, not np.allclose(limited, tau, atol=1e-8, rtol=0.)


@dataclass
class MITCommand:
    pos: np.ndarray
    vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray

    def total_torque(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        return self.tau + self.kp*(self.pos-q) + self.kd*(self.vel-qd)


class RSHardware:
    """复用分组读写、模式切换和生命周期；不在 demo 里另写电机协议。"""
    def __init__(self, robot: RebotArm, law: ImpedanceLaw, *, feedback_mode: str = 'motorbridge',
                 rate: float | None = None) -> None:
        self.robot, self.law = robot, law
        self.feedback_mode = feedback_mode
        self.rate = resolve_rate(feedback_mode, rate, control_space=law.control_space)
        self.requested_rate = self.rate
        self.feedback: MotorbridgeFeedback | SocketCANFeedback | None = None
        self.reporting_motors = []
        self.starting = True
        self.last_sample_times: np.ndarray | None = None
        self.feedback_age = 0.
        self.gripper_target = np.zeros(0)
        self.previous_tau: np.ndarray | None = None
        self.last_read_time = 0.
        self.last_read_cost = 0.
        self.last_send_cost = 0.
        self.last_send_attempt_cost = 0.
        self.cancel_reason = ''
        self.startup_estimator: VelocityEstimator | None = None
        self.startup_feedback_wait = 0.
        self.pre_enable_feedback_wait = 0.
        self.last_gripper_send = -np.inf
        self.cycle_periods: deque[float] = deque(maxlen=500)
        self.cancelled: Callable[[], bool] = lambda: False
        self.enable_attempted = False  # 使能可能只完成部分电机，必须在调用前记录。
        self.disable_unconfirmed = False
        self.disabled_for_preflight = False

    @property
    def control_timeout(self) -> float:
        return 3./self.rate if self.law.control_space == 'cartesian' else MAX_SAMPLE_DT

    def command_expired(self) -> bool:
        now = time.perf_counter()
        read_age = now-self.last_read_time
        if read_age > self.control_timeout:
            self.cancel_reason = (f'指令计算/发送过期: read_age={1000*read_age:.2f}ms '
                                  f'limit={1000*self.control_timeout:.2f}ms')
            return True
        if self.law.control_space == 'cartesian' and self.last_sample_times is not None:
            oldest = int(np.argmin(self.last_sample_times))
            age = now-float(self.last_sample_times[oldest])
            if age > self.control_timeout:
                self.cancel_reason = (f'{self.law.joint_names[oldest]} 反馈在计算/发送期间过期: '
                                      f'age={1000*age:.2f}ms read_age={1000*read_age:.2f}ms '
                                      f'limit={1000*self.control_timeout:.2f}ms')
                return True
        return False

    def read(self) -> np.ndarray:
        started = time.perf_counter()
        if self.feedback is None:
            q = self.robot.arm.get_positions(fresh=True, strict=True)
            self.last_sample_times = None
        else:
            snapshot = self.feedback.snapshot(max_age=MAX_SAMPLE_DT if self.starting else 3./self.rate)
            q, self.last_sample_times = snapshot.positions, snapshot.times
            self.feedback_age = float(np.max(time.perf_counter()-snapshot.times))
        self.law.validate_q(q)
        self.last_read_time = time.perf_counter()
        self.last_read_cost = self.last_read_time-started
        if self.last_read_cost > MAX_SAMPLE_DT:
            raise ControlFault(f'采样周期异常: 单次反馈读取耗时 {self.last_read_cost:.4f}s')
        return q

    def close(self) -> None:
        for motor in self.reporting_motors:
            try:
                motor.robstride_set_active_report(False)
            except Exception:
                pass  # 此处已请求失能，仍须继续关闭 SDK。
        self.reporting_motors.clear()
        if isinstance(self.feedback, SocketCANFeedback):
            self.feedback.close()

    def start_feedback(self, stop: Callable[[], bool]) -> None:
        self.require_disabled_probe()
        motors = [self.robot._motor_map[name] for name in self.law.joint_names]
        if self.feedback_mode == 'socketcan':
            cfg = load_cfg()
            joints = {joint.name: joint for joint in cfg['joints']}
            self.feedback = SocketCANFeedback(cfg['channel'], [joints[name] for name in self.law.joint_names])
            self.feedback.start()
        elif self.feedback_mode == 'query' or not MotorbridgeFeedback.supported(motors):
            return
        else:
            self.feedback = MotorbridgeFeedback(self.robot, self.law.joint_names)
        for motor in motors:
            self.reporting_motors.append(motor)
            motor.robstride_set_active_report(True)
        self.send_disabled_probe(stop)
        deadline = time.perf_counter()+1.
        while True:
            if stop():
                raise KeyboardInterrupt
            try:
                self.read()
                break
            except (FeedbackUnavailable, SocketFeedbackUnavailable) as exc:
                if time.perf_counter() >= deadline:
                    raise ControlFault(f'未收到完整的 {self.feedback_mode} 新状态反馈：{exc}；未使能电机') from exc
                self.send_disabled_probe(stop)
                time.sleep(.005)
        reference = self.robot.arm.get_positions(fresh=True, strict=True)
        if np.any(np.abs(self.read()-reference) > np.deg2rad(2.)):
            raise ControlFault('SDK 快照与 mechPos 相差超过 2°，请核对零位/SDK；未使能电机')
        self.feedback.reset_intervals()

    def send(self, command: MITCommand, total: np.ndarray) -> None:
        started = time.perf_counter()
        try:
            self.robot.arm.send_mit(command.pos, vel=command.vel, kp=command.kp,
                                    kd=command.kd, tau=command.tau, strict=True,
                                    cancelled=self.cancelled)
            if self.gripper_due():
                self.robot.gripper.send_mit(self.gripper_target, strict=True, cancelled=self.cancelled)
                self.last_gripper_send = time.perf_counter()
            if self.cancelled():
                raise InterruptedError('MIT 批次已取消，不提交完整发送记录')
            self.previous_tau = total.copy()
            self.last_send_cost = time.perf_counter()-started
        finally:
            self.last_send_attempt_cost = time.perf_counter()-started

    def finish_startup_feedback(self, q: np.ndarray, stop: Callable[[], bool]) -> np.ndarray:
        if self.feedback is None or self.law.control_space != 'cartesian':
            return q
        previous_times = self.last_sample_times.copy()
        checked = VelocityEstimator()
        checked.update(q, self.last_read_time, sample_times=previous_times)
        started = time.perf_counter()
        current = self.wait_feedback_advance(previous_times, stop, stage='启动反馈交接')
        checked.update(current, self.last_read_time, sample_times=self.last_sample_times)
        self.startup_estimator = checked
        self.startup_feedback_wait = time.perf_counter()-started
        return current

    def wait_feedback_advance(self, previous_times: np.ndarray, stop: Callable[[], bool],
                              *, stage: str) -> np.ndarray:
        deadline = time.perf_counter()+self.control_timeout
        detail = ', '.join(self.law.joint_names)
        while True:
            if stop():
                raise KeyboardInterrupt
            try:
                current = self.read()
                ages = time.perf_counter()-self.last_sample_times
                missing = ((self.last_sample_times <= previous_times) | (ages > 1./self.rate))
                if not np.any(missing) and time.perf_counter() <= deadline:
                    return current
                detail = ', '.join(self.law.joint_names[index] for index in np.flatnonzero(missing))
            except (FeedbackUnavailable, SocketFeedbackUnavailable) as exc:
                detail = str(exc)
            remaining = deadline-time.perf_counter()
            if remaining <= 0.:
                raise ControlFault(f'{stage}失败: {detail}; '
                                   f'wait_limit={1000*self.control_timeout:.2f}ms；未开始移动')
            time.sleep(min(.0002, remaining))

    def gripper_due(self) -> bool:
        return (self.robot.has_gripper and (self.law.control_space == 'joint'
                or time.perf_counter()-self.last_gripper_send >= 1./GRIPPER_REFRESH_RATE))

    def require_disabled_probe(self) -> None:
        if not self.disabled_for_preflight or self.disable_unconfirmed or self.enable_attempted:
            raise ControlFault('尚未通过严格失能调用或已尝试使能，禁止发送零增益测速指令')

    def send_disabled_probe(self, stop: Callable[[], bool]) -> None:
        self.require_disabled_probe()
        if stop():
            raise KeyboardInterrupt
        zeros = np.zeros(6)
        self.robot.arm.send_mit(zeros, vel=zeros, kp=zeros, kd=zeros, tau=zeros,
                                strict=True, cancelled=stop)

    def preflight(self, stop: Callable[[], bool]) -> TimingReport:
        self.require_disabled_probe()
        costs = []
        cycle_starts = []
        deadline = time.perf_counter()
        samples = max(PREFLIGHT_SAMPLES, int(np.ceil(self.rate))) if self.feedback else PREFLIGHT_SAMPLES
        for sample in range(PREFLIGHT_WARMUP+samples):
            if stop():
                raise KeyboardInterrupt
            if sample == PREFLIGHT_WARMUP and self.feedback is not None:
                self.feedback.reset_intervals()
            cycle_started = time.perf_counter()
            if sample >= PREFLIGHT_WARMUP:
                cycle_starts.append(cycle_started)
            self.read()
            tick = time.perf_counter()
            self.send_disabled_probe(stop)
            if self.gripper_due():
                n = self.robot.gripper.num_joints
                self.robot.gripper.send_mit(np.zeros(n), kp=np.zeros(n),
                                            kd=np.zeros(n), strict=True)
                self.last_gripper_send = time.perf_counter()
            write_cost = time.perf_counter()-tick
            # 干跑各控制分支、限幅/限速、MIT 拆分和最坏情况下的日志格式化。
            # 使用已验证的目标姿态做计算，实际写入始终为上面的零增益/零前馈。
            math_cost = benchmark_control_math(self, phase=CONTROL_PHASES[sample % len(CONTROL_PHASES)])
            if sample >= PREFLIGHT_WARMUP:
                costs.append([self.last_read_cost, write_cost, math_cost])
            deadline += 1./self.rate
            now = time.perf_counter()
            if now > deadline:
                deadline += (int((now-deadline)*self.rate)+1)/self.rate
            time.sleep(max(0., deadline-now))
        return TimingReport(np.array(costs), self.rate, np.diff(cycle_starts))

    def start(self, *, timing_only: bool = False,
              stop: Callable[[], bool] = lambda: False,
              report: Callable[[str], None] = print,
              before_enable: Callable[[np.ndarray], None] | None = None) -> np.ndarray | None:
        if (self.law.control_space == 'cartesian' and not timing_only
                and (self.requested_rate < 200. or self.feedback_mode == 'query')):
            raise ControlFault('笛卡尔阻抗配置未达标：运行要求 200..500Hz 高速反馈，query 仅用于测速；未连接硬件')
        self.robot.connect()
        self.startup_estimator = None
        self.startup_feedback_wait = 0.
        self.pre_enable_feedback_wait = 0.
        self.disabled_for_preflight = False
        self.disable_unconfirmed = True
        self.robot.disable_all(strict=True)
        self.disable_unconfirmed = False
        self.disabled_for_preflight = True
        if not self.robot.arm.mode_mit(kp=self.law.move_kp, kd=self.law.move_kd):
            raise ControlFault('机械臂切换 MIT 失败')
        if self.robot.has_gripper and not self.robot.gripper.mode_mit():
            raise ControlFault('夹爪切换 MIT 失败')
        self.start_feedback(stop)
        timing = self.preflight(stop)
        capacity = np.inf
        if self.feedback is None:
            feedback_info = 'motorbridge: fresh mechPos queries; native snapshot metadata unavailable or query selected'
        else:
            capacity = self.feedback.rate_capacity()
            feedback_info = self.feedback.describe()
        self.rate = timing.select_rate(capacity)
        ready = False
        if self.law.control_space == 'cartesian':
            ready = (self.feedback is not None and self.requested_rate >= 200.
                     and timing.max_cycle <= timing.budget
                     and timing.loop_capacity >= .95*self.requested_rate
                     and timing.max_period <= 1.5/self.requested_rate
                     and capacity >= .95*self.requested_rate)
            if self.feedback is not None:
                ready = ready and self.feedback.max_observed_period() <= 1.5/self.requested_rate
            if ready:
                self.rate = self.requested_rate
        report(timing.describe(self.rate))
        report('[FEEDBACK] '+feedback_info)
        if self.law.control_space == 'cartesian':
            report(f'[CARTESIAN RATE] requested={self.requested_rate:g}Hz ready={ready}; '
                   '要求有时间戳的反馈、主机/最慢轴速率>=95%、最大观测间隔<=1.5周期、工作开销<=80%周期')
            if not ready and not timing_only:
                raise ControlFault('笛卡尔阻抗通信/计算未达标，未使能；先运行 --timing-check 排查，'
                                   '可比较 --feedback motorbridge 与 --feedback socketcan')
            if timing_only:
                return None
        if self.rate < self.requested_rate:
            report(f'[RATE] 期望 {self.requested_rate:g}Hz，按通信/计算实测采用 '
                   f'{self.rate:g}Hz ({1000/self.rate:.2f}ms)；继续正常启动，不因未达期望频率退出。')
        if timing_only:
            return None
        self.starting = False
        if stop():
            raise KeyboardInterrupt
        if self.robot.has_gripper:
            self.gripper_target = self.robot.gripper.get_positions(fresh=True, strict=True)
        if self.feedback is not None and self.law.control_space == 'cartesian':
            previous_times = self.last_sample_times.copy()
            started = time.perf_counter()
            self.send_disabled_probe(stop)
            q = self.wait_feedback_advance(previous_times, stop, stage='使能前反馈同步')
            self.pre_enable_feedback_wait = time.perf_counter()-started
        else:
            q = self.read()
        if before_enable is not None:
            before_enable(q.copy())
        gravity = self.law.gravity(q)
        if np.any(np.abs(gravity) > self.law.torque_limit):
            raise ControlFault('启动重力前馈超过限幅')
        # 与 demo9 相同：先设真实位置目标，再使能，以电机内部 PD 保持。
        command = MITCommand(q, np.zeros(6), self.law.move_kp, self.law.move_kd, gravity)
        gravity, _ = self.law.limit_torque(gravity, None, 1./self.rate)
        self.last_gripper_send = -np.inf
        self.send(command, gravity)
        if stop():
            raise KeyboardInterrupt
        self.disabled_for_preflight = False  # 开始使能后，禁止再次运行零增益预检。
        self.enable_attempted = True
        self.robot.enable_all(strict=True)
        # 使能等待不属于控制采样周期。只使用使能后的新位置建立保持，
        # 初次主机速度取零，电机内部 kd 仍按实时速度提供阻尼；后续循环重新采样估速。
        q = self.read()
        qd, dt = np.zeros(6), 1./self.rate
        holder = CartesianImpedanceController(self.law, q, self.last_read_time, rate=self.rate)
        holder.measured_velocity = qd
        tau = self.law.joint_hold(q, qd, q, np.zeros(6))
        tau, _ = self.law.limit_torque(tau, self.previous_tau, dt)
        command = holder.mit_command(q, tau)
        self.send(command, command.total_torque(q, qd))
        return self.finish_startup_feedback(q, stop)


class CartesianImpedanceController:
    def __init__(self, law: ImpedanceLaw, q_start: np.ndarray, now: float,
                 *, rate: float = CONTROL_RATE) -> None:
        self.law, self.q_start = law, q_start.copy()
        self.started = now
        self.move_plan = JointMotion.plan(q_start, law.initial, max_speed=MAX_MOVE_SPEED,
                                         min_duration=MIN_MOVE_DURATION,
                                         max_acceleration=MAX_MOVE_ACCELERATION)
        self.duration = START_HOLD_DURATION+self.move_plan.duration
        self.estimator = VelocityEstimator()
        self.rate_monitor = ControlRateMonitor(rate)
        self.feedback_monitors = [ControlRateMonitor(rate, label=f'{name}反馈') for name in law.joint_names]
        self.phase = 'MOVE'
        self.quiet_since: float | None = None
        self.blend_started: float | None = None
        self.home_started: float | None = None
        self.home_start: np.ndarray | None = None
        self.home_duration = 0.
        self.home_plan: JointMotion | None = None
        self.home_quiet_since: float | None = None
        self.hold_position: np.ndarray | None = None
        self.hold_reason = ''
        self.last_log = -np.inf
        self.last_logged_phase = ''
        self.measured_velocity = np.zeros(6)
        self.servo = (q_start.copy(), np.zeros(6), law.move_kp.copy(), law.move_kd.copy())

    def mit_command(self, q: np.ndarray, limited_total: np.ndarray) -> MITCommand:
        """移动时拆分内环 PD 和前馈；笛卡尔阶段零增益直接发送限幅后的力矩。"""
        if (limited_total.shape != (6,) or not np.all(np.isfinite(limited_total))
                or np.any(np.abs(limited_total) > self.law.torque_limit+TORQUE_EPSILON)):
            raise ControlFault('MIT 目标总力矩无效或已超限')
        pos, vel, kp, kd = self.servo
        pd = kp*(pos-q) + kd*(vel-self.measured_velocity)
        # 不可单独裁剪前馈，否则会改变已经限幅/限速的总力矩。
        # 每关节选最大的 s∈[0,1]，满足 |limited_total-s*pd| <= torque_limit。
        # 同比缩小 kp/kd，保留内环 PD 的方向；不修改 self.servo 中的增益。
        room = np.maximum(0., self.law.torque_limit+np.sign(pd)*limited_total)
        scale = np.ones(6)
        np.divide(room, np.abs(pd), out=scale, where=np.abs(pd) > room)
        kp, kd = kp*scale, kd*scale
        ff = limited_total - (kp*(pos-q) + kd*(vel-self.measured_velocity))
        command = MITCommand(pos.copy(), vel.copy(), kp.copy(), kd.copy(), ff)
        # MIT 内环依赖电机实时速度；这里的总力矩仅为本次反馈下的估计。
        total = command.total_torque(q, self.measured_velocity)
        if (not np.all(np.isfinite(ff))
                or np.any(np.abs(ff) > self.law.torque_limit+TORQUE_EPSILON)
                or not np.allclose(total, limited_total, atol=TORQUE_EPSILON, rtol=0.)):
            raise ControlFault('MIT 指令无法保持限幅/限速后的估计总力矩')
        return command

    def request_home(self, q: np.ndarray, now: float) -> None:
        """从中断时的实际角度开始归零，不从初始轨迹或笛卡尔目标跳转。"""
        self.law.validate_q(q)
        if self.home_started is not None:
            raise ControlFault("已经在归零")
        # 独立调用也要接纳交接样本，保留速度历史；不能跳过它后再差分两批反馈。
        if self.estimator.previous_time != now:
            self.estimator.update(q, now)
        elif not np.array_equal(q, self.estimator.previous):
            raise ControlFault("同一采样时间的位置样本不一致")
        self._begin_home(q, now)

    def _begin_home(self, q: np.ndarray, now: float) -> None:
        self.hold_position = None
        self.hold_reason = ''
        self.home_start = q.copy()
        self.home_plan = JointMotion.plan(q, self.law.home, max_speed=MAX_MOVE_SPEED,
                                         min_duration=MIN_MOVE_DURATION,
                                         max_acceleration=MAX_MOVE_ACCELERATION)
        self.home_duration = self.home_plan.duration
        self.home_started = now
        self.home_quiet_since = None
        self.phase = 'HOMING'

    def joint_status(self, q: np.ndarray) -> str:
        ref = self.servo[0]
        error = np.rad2deg(ref-q)
        worst = self.law.joint_names[int(np.argmax(np.abs(error)))]
        def fmt(values):
            return np.array2string(values, precision=3, separator=',', max_line_width=1000)
        return (f'worst={worst} q_deg={fmt(np.rad2deg(q))} '
                f'ref_deg={fmt(np.rad2deg(ref))} error_deg={fmt(error)} '
                f'qd_rad_s={fmt(self.measured_velocity)}')

    def _hold(self, q: np.ndarray, qd: np.ndarray, dt: float) -> tuple[np.ndarray, float, str]:
        self.phase = 'PROTECTED_HOLD'
        zeros = np.zeros(6)
        self.servo = (self.hold_position, zeros, self.law.move_kp, self.law.move_kd)
        tau = self.law.joint_hold(q, qd, self.hold_position, zeros)
        return tau, dt, f'暂停轨迹并保持位置: {self.hold_reason}'

    def capture_hold(self, q: np.ndarray, now: float, *, sample_times: np.ndarray | None,
                     dt: float, estimator: VelocityEstimator | None = None) -> tuple[np.ndarray, float, str]:
        """沿用已经核对的恢复样本和速度历史，捕获当前位置，不追赶暂停轨迹。"""
        self.law.validate_q(q)
        checked = self.estimator if estimator is None else estimator
        if checked.previous_time != now:
            checked.update(q, now, sample_times=sample_times)
        elif (not np.array_equal(q, checked.previous)
              or (sample_times is not None and not np.array_equal(sample_times, checked.sample_times))):
            raise ControlFault('保持交接样本与已核对的反馈不一致')
        self.estimator = checked
        self.measured_velocity = checked.filtered.copy()
        self.hold_position = q.copy()
        self.hold_reason = '控制更新短暂延迟，已恢复新鲜反馈；暂停并保持当前位置'
        return self._hold(q, self.measured_velocity, dt)

    def step(self, q: np.ndarray, now: float, *, home_requested: bool = False,
             sample_times: np.ndarray | None = None
             ) -> tuple[np.ndarray, float, str]:
        self.law.validate_q(q)
        previous_stamps = self.estimator.sample_times
        qd, dt = self.estimator.update(q, now, sample_times=sample_times)
        self.measured_velocity = qd.copy()
        try:
            if self.law.control_space == 'cartesian' and self.hold_position is None:
                self.rate_monitor.observe(dt)
                if sample_times is not None and previous_stamps is not None:
                    for monitor, elapsed in zip(self.feedback_monitors, self.estimator.sample_times-previous_stamps):
                        if elapsed > 0.:
                            monitor.observe(float(elapsed))
            return self._phase_step(q, qd, dt, now, home_requested=home_requested)
        except HoldRequest as exc:
            # 只接纳有效反馈之后的可恢复控制误差；通信/限位/速度异常仍向外抛出。
            self.hold_position = q.copy()
            self.hold_reason = str(exc)
            return self._hold(q, qd, dt)

    def _phase_step(self, q: np.ndarray, qd: np.ndarray, dt: float, now: float,
                    *, home_requested: bool) -> tuple[np.ndarray, float, str]:
        if home_requested and self.home_started is None:
            self._begin_home(q, now)
        if self.hold_position is not None:
            return self._hold(q, qd, dt)
        if self.home_started is not None:
            elapsed = now-self.home_started
            ref, ref_vel = self.home_plan.sample(elapsed)
            tau = self.law.joint_hold(q, qd, ref, ref_vel)
            self.servo = (ref, ref_vel, self.law.move_kp, self.law.move_kd)
            home_error = float(np.max(np.abs(self.law.home-q)))
            if elapsed >= self.home_duration:
                self.phase = 'HOME_SETTLE'
                if home_error < ARRIVAL_TOLERANCE and np.max(np.abs(qd)) < .05:
                    if self.home_quiet_since is None:
                        self.home_quiet_since = now
                    elif now-self.home_quiet_since >= SETTLE_TIME:
                        self.phase = 'HOME_HOLD'
                else:
                    self.home_quiet_since = None
                if (elapsed > self.home_duration+SETTLE_TIMEOUT
                        and self.phase != 'HOME_HOLD'):
                    self.phase = 'HOME_WAIT'
            info = f'max_home_error={np.rad2deg(home_error):.2f}deg'
            if self.phase == 'HOME_WAIT':
                info += ' 归零到位超时，继续保持零位目标；尚未确认到位'
            return tau, dt, info
        elapsed = now-self.started
        if elapsed < self.duration:
            ref, ref_vel = self.move_plan.sample(elapsed-START_HOLD_DURATION)
            tau = self.law.joint_hold(q, qd, ref, ref_vel)
            self.servo = (ref, ref_vel, self.law.move_kp, self.law.move_kd)
            return tau, dt, f'max_joint_error={np.rad2deg(np.max(np.abs(ref-q))):.2f}deg'
        if self.blend_started is None:
            self.phase = 'SETTLE'
            arrived = (np.max(np.abs(q-self.law.initial)) < ARRIVAL_TOLERANCE
                       and np.max(np.abs(qd)) < .05)
            if arrived:
                if self.quiet_since is None:
                    self.quiet_since = now
                elif now-self.quiet_since >= SETTLE_TIME:
                    self.blend_started = now
            else:
                self.quiet_since = None
            if elapsed > self.duration+SETTLE_TIMEOUT and self.blend_started is None:
                self.phase = 'SETTLE_WAIT'
            if self.blend_started is None:
                self.servo = (self.law.initial, np.zeros(6), self.law.move_kp, self.law.move_kd)
                info = ('初始姿态到位超时，继续保持目标，尚未进入阻抗；Ctrl+C 可归零'
                        if self.phase == 'SETTLE_WAIT' else 'waiting for arrival')
                return self.law.joint_hold(q, qd, self.law.initial, np.zeros(6)), dt, info
        alpha, _ = minimum_jerk_profile(now-self.blend_started, BLEND_DURATION)
        cart = self.law.cartesian(q, qd)
        if alpha < 1.:
            self.phase = 'BLEND'
            hold = self.law.joint_hold(q, qd, self.law.initial, np.zeros(6))
            tau = (1.-alpha)*hold + alpha*cart.tau
        else:
            self.phase = 'IMPEDANCE'
            tau = cart.tau
        if self.law.control_space == 'cartesian':
            self.servo = (self.law.initial, np.zeros(6),
                          (1.-alpha)*self.law.move_kp, (1.-alpha)*self.law.move_kd)
        else:
            self.servo = (self.law.initial, np.zeros(6),
                          (1.-alpha)*self.law.move_kp+alpha*self.law.hold_kp,
                          (1.-alpha)*self.law.move_kd+alpha*self.law.hold_kd)
        info = (f'pos_error={1000*np.linalg.norm(cart.error[:3]):.2f}mm '
                f'rot_error={np.rad2deg(np.linalg.norm(cart.error[3:])):.2f}deg')
        return tau, dt, info


def format_status(controller: CartesianImpedanceController, hardware: RSHardware, q: np.ndarray,
                  tau: np.ndarray, dt: float, info: str, saturated: bool) -> str:
    age_info = (f'{1000*hardware.feedback_age:.1f}ms'
                if hardware.last_sample_times is not None else 'n/a(query)')
    periods = np.asarray(hardware.cycle_periods if hardware.cycle_periods else [dt])
    loop_hz = 1./float(np.mean(periods))
    late = int(np.count_nonzero(periods > 1.5/hardware.rate))
    return (f'[{controller.phase}] {info} dt={1000*dt:.2f}ms actual_hz={1./dt:.1f} '
            f'loop_hz={loop_hz:.1f} dt_max={1000*np.max(periods):.2f}ms late={late}/{len(periods)} '
            f'read={1000*hardware.last_read_cost:.1f}ms '
            f'write={1000*hardware.last_send_cost:.1f}ms '
            f'feedback_age={age_info} '
            f'limited={saturated} estimated_tau={np.round(tau, 3)} '
            f'{controller.joint_status(q)}')


def benchmark_control_math(hardware: RSHardware, *, phase: str | None = None) -> float:
    """Measure one complete dry-run branch, or the worst of all branches; never transmits."""
    law, q = hardware.law, hardware.law.initial
    worst = 0.
    for phase in CONTROL_PHASES if phase is None else (phase,):
        tick = time.perf_counter()
        controller = CartesianImpedanceController(law, q, tick, rate=hardware.rate)
        controller.rate_monitor.intervals.extend([1./hardware.rate]*49)
        for monitor in controller.feedback_monitors:
            monitor.intervals.extend([1./hardware.rate]*49)
        # 用独立的合成时刻推进估计器，不将测速计算写回真实反馈或力矩历史。
        dt = 1./hardware.rate
        controller.estimator.update(q, tick, sample_times=np.full(6, tick))
        if phase == 'SETTLE':
            controller.started -= controller.duration
        elif phase in ('BLEND', 'IMPEDANCE'):
            controller.started -= controller.duration
            controller.blend_started = tick-(.5 if phase == 'BLEND' else 1.)*BLEND_DURATION
        elif phase == 'HOMING':
            controller._begin_home(q, tick)
        elif phase == 'PROTECTED_HOLD':
            controller.hold_position = q.copy()
        tau, dt, info = controller.step(q, tick+dt, sample_times=np.full(6, tick+dt))
        tau, saturated = law.limit_torque(tau, np.zeros(6), dt)
        command = controller.mit_command(q, tau)
        command.total_torque(q, controller.measured_velocity)
        format_status(controller, hardware, q, tau, dt, info, saturated)
        worst = max(worst, time.perf_counter()-tick)
    return worst


def print_setup(law: ImpedanceLaw, *, feedback_mode: str = 'motorbridge', rate: float | None = None) -> None:
    print('Demo11: MIT 最小 jerk 到初始姿态 ->', law.control_space, '阻抗保持')
    print('initial joints (deg):', np.rad2deg(law.initial))
    print('Ctrl+C home joints (deg):', np.rad2deg(law.home))
    print('EE frame:', law.frame_name)
    print('target XYZ (m):', law.target.translation)
    print('target RPY (deg):', np.rad2deg(pin.rpy.matrixToRpy(law.target.rotation)))
    print('gripper forward (+X in base):', np.round(law.target.rotation[:, 0], 6))
    print('move/home motor KP (Demo11 profile):', law.move_kp, 'KD:', law.move_kd)
    if law.control_space == 'cartesian':
        print('Cartesian stiffness [N/m, N*m/rad]:', law.cartesian_stiffness)
        print('Cartesian damping [N*s/m, N*m*s/rad]:', law.cartesian_damping)
        print('IMPEDANCE motor KP=KD=0; coriolis compensation:', law.compensate_coriolis)
    else:
        print('impedance hold motor KP:', law.hold_kp, 'KD:', law.hold_kd)
    print('gravity scale (cartesian_impedance):', law.tau_scale)
    print('startup hold (s):', START_HOLD_DURATION, 'move speed (rad/s):', MAX_MOVE_SPEED,
          'acceleration (rad/s^2):', MAX_MOVE_ACCELERATION)
    print('feedback mode:', feedback_mode, 'host target rate (Hz):',
          resolve_rate(feedback_mode, rate, control_space=law.control_space))
    print('estimated total torque limits (N*m):', law.torque_limit)


def run(law: ImpedanceLaw, *, timing_only: bool = False, feedback_mode: str = 'motorbridge',
        rate: float | None = None, controller_factory=None, sample_observer=None,
        before_enable: Callable[[np.ndarray], None] | None = None,
        home_on_interrupt: bool = True) -> None:
    robot = RebotArm()
    hardware = RSHardware(robot, law, feedback_mode=feedback_mode, rate=rate)
    stop = StopRequest()
    previous_handler = signal.signal(signal.SIGINT, stop.handle)
    logger: BackgroundLog | None = None
    enter_key: EnterKey | None = None
    shutting_down = threading.Event()
    hardware.cancelled = shutting_down.is_set
    faults = RuntimeFault()
    recovery_samples = RecoverySamples()
    explicit_release = False
    last_fault_notice = -np.inf
    observer_failed = False

    def log(message: str) -> None:
        try:
            if logger is not None:
                logger.post(message)
        except Exception:
            pass  # 日志失败不能撤销电机保持。

    def fault_notice(*, force: bool = False) -> None:
        nonlocal last_fault_notice
        now = time.perf_counter()
        if faults.pending.is_set() and (force or now-last_fault_notice >= 2.):
            last_fault_notice = now
            if hardware.disable_unconfirmed:
                log(f'[FAULT_WAIT] 启动失能未确认: {faults.reason}；'
                    '已停止模式切换和测速，不发新 MIT 指令，不重试失能或关闭 SDK。'
                    '托稳后按 Enter 或第二次 Ctrl+C 明确退出。')
                return
            log(f'[FAULT_WAIT] {faults.reason}；暂停轨迹，不自动失能。'
                '反馈恢复并通过检查后保持当前位置；无可信反馈时不发新指令，'
                '电机端保留最后收到的指令（不保证失联时能保持）。托稳后按 Enter 或第二次 Ctrl+C 失能。')

    def request_hold(reason: str) -> None:
        if faults.request(reason):
            fault_notice(force=True)

    def wait_for_release() -> None:
        # 启动使能后的异常或监督线程异常也不能直接落入 disconnect。
        while stop.count < 2:
            try:
                fault_notice()
                if enter_key is not None and enter_key.poll():
                    return
                time.sleep(.05)
            except BaseException:
                log('[FAULT_WAIT] 仍等待人工托稳后明确退出，不自动失能。')

    try:
        # 所有同步终端输出、日志线程启动都在使能之前完成。
        if not timing_only:
            print(('到位超时继续保持；Ctrl+C 平滑归零。' if home_on_interrupt else
                   'Ctrl+C 停止采集并保持当前位置，不自动归零。')+
                  '归零到位/超时、保护保持或故障等待时，确认支撑后按 Enter 失能；'
                  '再次 Ctrl+C 失能。运行故障不自动失能；'
                  '无可信反馈时保留电机端最后指令并等待人工接管。')
        else:
            print('[TIMING] 仅测速，电机保持失能。')
        logger = BackgroundLog()
        enter_key = EnterKey(sys.stdin)
        start_options = dict(timing_only=timing_only, stop=stop.requested)
        if before_enable is not None:
            start_options['before_enable'] = before_enable
        q_start = hardware.start(**start_options)
        if timing_only:
            print('[TIMING DONE] 测速结束，未使能电机。')
            return
        hardware.enable_attempted = True
        factory = controller_factory or CartesianImpedanceController
        controller = factory(law, q_start, hardware.last_read_time, rate=hardware.rate)
        if hardware.startup_estimator is not None:
            controller.estimator = hardware.startup_estimator
        else:
            controller.estimator.update(q_start, hardware.last_read_time,
                                        sample_times=hardware.last_sample_times)
        log(f'[MOVE] duration={controller.duration:.2f}s; 期望={hardware.requested_rate:g}Hz '
            f'运行={hardware.rate:g}Hz feedback={hardware.feedback_mode} '
            f'pre_enable_sync={1000*hardware.pre_enable_feedback_wait:.2f}ms '
            f'startup_sync={1000*hardware.startup_feedback_wait:.2f}ms '
            f'feedback_age={1000*hardware.feedback_age:.2f}ms')
        home_ready = threading.Event()
        watchdog = LoopWatchdog(hardware.rate, control_timeout=hardware.control_timeout)

        def needs_recovery(now: float) -> bool:
            return (watchdog.hold_requested.is_set()
                    or now-controller.estimator.previous_time > hardware.control_timeout)

        def control_step(r: RebotArm, nominal_dt: float) -> None:
            nonlocal observer_failed
            # 与 demo9/10 一样复用 RebotArm 的后台控制循环；仅主线程处理终端。
            if shutting_down.is_set() or stop.count >= 2:
                return
            if not home_on_interrupt and stop.requested() and controller.hold_position is None:
                request_hold('用户停止采集；保持当前位置，不恢复扫描、不自动归零')
            generation = faults.generation
            hardware.cancel_reason = ''

            def cancelled() -> bool:
                if shutting_down.is_set() or stop.count >= 2:
                    hardware.cancel_reason = '用户请求退出'
                    return True
                if not home_on_interrupt and stop.requested() and controller.hold_position is None:
                    request_hold('用户停止采集；保持当前位置，不恢复扫描、不自动归零')
                if generation != faults.generation:
                    hardware.cancel_reason = '监督线程已请求保护，旧批次取消'
                    return True
                return hardware.command_expired()

            hardware.cancelled = cancelled
            try:
                watchdog.set_stage('读取反馈')
                q = hardware.read()
                now = hardware.last_read_time
                if hardware.cancelled():
                    return
                beginning_home = home_on_interrupt and stop.requested() and controller.home_started is None
                watchdog.set_stage('控制计算')
                recovering = faults.pending.is_set()
                if recovering:
                    if not recovery_samples.ready(q, now, hardware.last_sample_times,
                                                  reference=controller.estimator):
                        return
                    reason = faults.reason
                    tau, dt, info = controller.capture_hold(
                        q, now, sample_times=hardware.last_sample_times, dt=1./hardware.rate,
                        estimator=recovery_samples.estimator)
                    controller.hold_reason = reason
                    info = f'故障后保持当前位置: {reason}'
                    beginning_home = False
                elif needs_recovery(now):
                    request_hold('控制更新间隔过长，等待两批可信新反馈后保持')
                    return
                else:
                    tau, dt, info = controller.step(q, now, home_requested=beginning_home,
                                                  sample_times=hardware.last_sample_times)
                checked_samples = recovery_samples.estimator if recovering else controller.estimator
                # 计算本身也可能暂停；恢复后再读一次，不能下发暂停前算出的旧指令。
                if (watchdog.hold_requested.is_set()
                        or hardware.command_expired()):
                    q = hardware.read()
                    now = hardware.last_read_time
                    checked_samples.update(q, now, sample_times=hardware.last_sample_times)
                    tau, dt, info = controller.capture_hold(
                        q, now, sample_times=hardware.last_sample_times, dt=1./hardware.rate,
                        estimator=checked_samples)
                    beginning_home = False
                tau, saturated = law.limit_torque(tau, hardware.previous_tau, dt)
                command = controller.mit_command(q, tau)
                # MIT 拆分也属于计算阶段；它恢复后同样不能发送旧反馈下的指令。
                if (watchdog.hold_requested.is_set()
                        or hardware.command_expired()):
                    q = hardware.read()
                    now = hardware.last_read_time
                    checked_samples.update(q, now, sample_times=hardware.last_sample_times)
                    tau, dt, info = controller.capture_hold(
                        q, now, sample_times=hardware.last_sample_times, dt=1./hardware.rate,
                        estimator=checked_samples)
                    beginning_home = False
                    tau, saturated = law.limit_torque(tau, hardware.previous_tau, dt)
                    command = controller.mit_command(q, tau)
                    if hardware.command_expired():
                        return  # 计算仍持续阻塞，交给独立心跳监督；不下发旧指令。
                if hardware.cancelled():
                    return
                watchdog.set_stage('发送 MIT')
                hardware.send(command, command.total_torque(q, controller.measured_velocity))
                hardware.cycle_periods.append(dt)
                if recovering:
                    faults.complete(generation)
                    recovery_samples.reset()
                if time.perf_counter()-now <= hardware.control_timeout:
                    watchdog.complete()
                if sample_observer is not None and not observer_failed:
                    try:
                        sample_observer(hardware, controller, q, command, dt, saturated)
                    except Exception:
                        observer_failed = True
                        raise
                if beginning_home and controller.phase in ('HOMING', 'HOME_SETTLE', 'HOME_HOLD', 'HOME_WAIT'):
                    log(f'[HOMING] 平滑归零，时长 {controller.home_duration:.2f}s；'
                        '再次 Ctrl+C 退出并失能。')
                phase_changed = controller.phase != controller.last_logged_phase
                if phase_changed or now-controller.last_log >= .5:
                    controller.last_log = now
                    controller.last_logged_phase = controller.phase
                    log(format_status(controller, hardware, q, tau, dt, info, saturated))
                if controller.phase in ('HOME_HOLD', 'HOME_WAIT', 'PROTECTED_HOLD'):
                    if phase_changed or not home_ready.is_set():
                        if controller.phase == 'HOME_HOLD':
                            message = '已在容差内归零并保持。'
                        elif controller.phase == 'HOME_WAIT':
                            message = '归零到位超时，继续保持零位目标；尚未确认到位。'
                        else:
                            message = f'已暂停轨迹，保持捕获位置：{controller.hold_reason}。'
                        log(f'[{controller.phase}] {message}托稳后按 Enter 失能断开；'
                            +('Ctrl+C 请求归零（若已请求过则退出失能）。' if home_on_interrupt else
                              '第二次 Ctrl+C 退出失能。'))
                    home_ready.set()
                else:
                    home_ready.clear()
            except Exception as exc:
                recovery_samples.reset()
                if shutting_down.is_set() or stop.count >= 2:
                    return
                detail = f'; {hardware.cancel_reason}' if hardware.cancel_reason else ''
                request_hold(f'{exc}{detail} (read={1000*hardware.last_read_cost:.2f}ms, '
                             f'write_attempt={1000*hardware.last_send_attempt_cost:.2f}ms, '
                             f'previous_write={1000*hardware.last_send_cost:.2f}ms)')

        robot.start_control_loop(control_step, rate=hardware.rate)
        input_warning_posted = False
        while stop.count < 2:
            if not home_on_interrupt and stop.requested() and controller.hold_position is None:
                request_hold('用户停止采集；保持当前位置，不恢复扫描、不自动归零')
            try:
                if watchdog.poll():
                    request_hold('控制更新延迟，恢复后捕获新位置保持，不追赶原轨迹')
            except ControlFault as exc:
                request_hold(str(exc))  # 心跳超时也等待人工接管，不自动失能或关闭 SDK。
            fault_notice()
            # 运动期间也消费输入，防止此前的 Enter 排队到保持阶段才触发失能。
            ready_for_release = home_ready.is_set() or faults.pending.is_set()
            if enter_key.poll() and ready_for_release:
                explicit_release = True
                break
            if ready_for_release and not enter_key.available and not input_warning_posted:
                log('[HOLD] 标准输入已关闭或不可用，继续等待；托稳后第二次 Ctrl+C 失能断开。')
                input_warning_posted = True
            time.sleep(min(.01, 1./hardware.rate) if law.control_space == 'cartesian' else .01)
        explicit_release = True  # Enter 或第二次 Ctrl+C 是运行中仅有的退出条件。
    except BaseException as exc:
        if not (hardware.enable_attempted or hardware.disable_unconfirmed):
            raise
        request_hold(f'启动使能后或监督线程异常: {exc}')
        wait_for_release()
        explicit_release = True
    finally:
        if (hardware.enable_attempted or hardware.disable_unconfirmed) and not explicit_release:
            request_hold('退出路径尚未得到人工失能请求')
            wait_for_release()
        shutting_down.set()
        try:
            # 运行中仅在人工托稳后明确退出才到这里；阻止恢复后的旧批次继续发送。
            robot.disconnect(before_close=hardware.close, control_timeout=.05)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            if enter_key is not None:
                enter_key.close()
            if logger is not None:
                logger.close()
        print('[DISCONNECTED] 已请求电机失能并断开。')

