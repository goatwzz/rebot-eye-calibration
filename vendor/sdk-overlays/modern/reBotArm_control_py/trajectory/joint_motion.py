"""关节移动及归零共用的最小 jerk 时间剖面。纯计算，不连接硬件。"""
from dataclasses import dataclass

import numpy as np


def minimum_jerk_profile(elapsed: float, duration: float) -> tuple[float, float]:
    """Position blend and its time derivative; zero duration means arrived."""
    if not np.isfinite(elapsed) or not np.isfinite(duration) or duration < 0:
        raise ValueError('轨迹时间必须有限，duration 不得为负')
    if duration == 0:
        return 1., 0.
    s = float(np.clip(elapsed/duration, 0., 1.))
    return 10*s**3-15*s**4+6*s**5, (30*s**2-60*s**3+30*s**4)/duration


@dataclass
class JointMotion:
    start: np.ndarray
    target: np.ndarray
    duration: float

    @classmethod
    def plan(cls, start, target, *, max_speed: float, min_duration: float = 0.,
             max_acceleration: float | None = None):
        start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
        if (start.shape != target.shape or start.ndim != 1 or start.size == 0
                or not np.all(np.isfinite(start)) or not np.all(np.isfinite(target))
                or not np.isfinite(max_speed) or max_speed <= 0
                or not np.isfinite(min_duration) or min_duration < 0
                or (max_acceleration is not None
                    and (not np.isfinite(max_acceleration) or max_acceleration <= 0))):
            raise ValueError('无效的关节轨迹起点、终点或速度/时间限制')
        duration = max(min_duration, 1.875*float(np.max(np.abs(target-start)))/max_speed)
        if max_acceleration is not None:
            # Quintic s'' 的峰值为 (10/sqrt(3))/T²。
            duration = max(duration, float(np.sqrt(
                (10./np.sqrt(3.))*np.max(np.abs(target-start))/max_acceleration)))
        return cls(start.copy(), target.copy(), duration)

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        if elapsed >= self.duration:
            return self.target.copy(), np.zeros_like(self.target)
        if elapsed <= 0:
            return self.start.copy(), np.zeros_like(self.start)
        blend, speed = minimum_jerk_profile(elapsed, self.duration)
        return self.start+(self.target-self.start)*blend, (self.target-self.start)*speed
