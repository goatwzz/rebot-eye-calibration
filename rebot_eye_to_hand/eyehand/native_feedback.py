"""Task sampling via upstream MotorBridge's public typed snapshot API.

No socket, ctypes, native structures or protocol decoding. Ages/sequences come
from Motor.get_state_snapshot(), not from timestamps invented for cached reads.
"""
from collections import deque
from types import SimpleNamespace
import time
import numpy as np


class FeedbackUnavailable(ValueError):
    pass


class NativeFeedback:
    @staticmethod
    def supported(motors):
        return all(callable(getattr(m,'get_state_snapshot',None)) for m in motors)

    def __init__(self,robot,names):
        self.robot=robot;self.names=list(names)
        self.motors=[robot._motor_map[name] for name in names]
        if not self.supported(self.motors):raise RuntimeError('Upstream MotorBridge snapshot API unavailable')
        self.q=np.zeros(len(names));self.times=np.full(len(names),-np.inf)
        self.sequences=np.zeros(len(names),dtype=np.int64)
        self.velocities=np.full(len(names),np.nan);self.torques=np.full(len(names),np.nan)
        self.intervals=[deque(maxlen=2048) for _ in names]
        self.timing_started=None;self.timing_reads=0
        self.timing_anchor=np.zeros(len(names),dtype=bool)
        self.first_update=[None for _ in names];self.worst_gap=[None for _ in names]
        self.max_ages=np.zeros(len(names))
        self.timing_times=np.full(len(names),-np.inf)
        self.timing_sequences=np.zeros(len(names),dtype=np.int64)
        self.timestamp_bound_adjustments=np.zeros(len(names),dtype=np.int64)
        self.last_timestamp_bound_adjustment=[None for _ in names]

    def snapshot(self,max_age):
        self.timing_reads+=1
        limits=np.broadcast_to(np.asarray(max_age,float),self.q.shape)
        if not np.isfinite(limits).all() or np.any(limits<=0):raise ValueError('Invalid feedback age limits')
        for controller in self.robot._ctrl_map.values():controller.poll_feedback_once()
        for i,motor in enumerate(self.motors):
            before=time.perf_counter();snapshot=motor.get_state_snapshot()
            after=time.perf_counter()
            if snapshot is None:raise FeedbackUnavailable(self.names[i]+' has no upstream snapshot')
            state=snapshot.state;age=snapshot.feedback_age_s;sequence=snapshot.feedback_sequence
            if state.status_code:raise RuntimeError(f'{self.names[i]} motor fault 0x{state.status_code:02x}')
            if (not np.isfinite([state.pos,age]).all() or age<0 or sequence<=0):
                raise RuntimeError(self.names[i]+' invalid upstream feedback')
            if age>limits[i]:raise FeedbackUnavailable(
                f'{self.names[i]} upstream feedback expired: age={1000*age:.2f}ms, limit={1000*limits[i]:.1f}ms')
            if self.timing_anchor[i]:self.max_ages[i]=max(self.max_ages[i],age)
            if sequence<self.sequences[i]:raise RuntimeError(self.names[i]+' feedback sequence reversed')
            if sequence==self.sequences[i]:
                if state.pos!=self.q[i]:raise RuntimeError(self.names[i]+' repeated sequence changed position')
                continue
            received_at=before-age
            # The SDK age is sampled INSIDE the API call, not at `before`.
            # Therefore receive time lies in [before-age, after-age]. Only
            # disjoint reversed bounds contradict an increasing native sequence.
            if after-age<self.times[i]:
                raise RuntimeError(self.names[i]+' feedback time reversed (non-overlapping API time bounds)')
            if received_at<=self.times[i]:
                self.timestamp_bound_adjustments[i]+=1
                self.last_timestamp_bound_adjustment[i]=dict(
                    previous_sequence=int(self.sequences[i]),sequence=int(sequence),
                    api_call_ms=1000*(after-before),sdk_age_ms=1000*age,
                    lower_bound_regression_ms=1000*(self.times[i]-received_at))
                # Still a conservative lower bound: a newer sequence cannot
                # predate the prior receive time. Cached reads never enter here.
                received_at=self.times[i]
            in_window=self.timing_started is None or received_at>=self.timing_started
            if self.timing_anchor[i] and in_window:
                # Average new-frame period; skipped frames cannot prove worst
                # packet gap. Host-observed new-snapshot gaps are also checked.
                gap=received_at-self.timing_times[i];count=int(sequence-self.timing_sequences[i])
                if gap>0:
                    self.intervals[i].append((gap,count))
                    self.timing_times[i]=received_at;self.timing_sequences[i]=sequence
                    if self.worst_gap[i] is None or gap>self.worst_gap[i]['gap_s']:
                        self.worst_gap[i]=dict(gap_s=gap,sequence_delta=count,read_index=self.timing_reads,
                            observed_elapsed_s=None if self.timing_started is None else before-self.timing_started)
            elif in_window:
                self.timing_anchor[i]=True;self.max_ages[i]=age
                self.timing_times[i]=received_at;self.timing_sequences[i]=sequence
                self.first_update[i]=dict(read_index=self.timing_reads,
                    observed_elapsed_s=None if self.timing_started is None else before-self.timing_started)
            # Native MotorState values from the SAME snapshot, no extra I/O.
            # Missing/nonfinite optional diagnostics do not invent velocity.
            for field,values in [('vel',self.velocities),('torq',self.torques)]:
                value=getattr(state,field,float('nan'))
                values[i]=value if np.isfinite(value) else float('nan')
            self.q[i]=state.pos;self.times[i]=received_at;self.sequences[i]=sequence
        if np.any(time.perf_counter()-self.times>limits):raise FeedbackUnavailable('Upstream snapshot stream expired')
        return SimpleNamespace(positions=self.q.copy(),times=self.times.copy(),sequences=self.sequences.copy(),
            velocities=self.velocities.copy(),torques=self.torques.copy())

    def reset_intervals(self):
        for intervals in self.intervals:intervals.clear()
        # Measurement needs TWO new observations in this window. Keep the
        # real feedback caches/ages/sequences; discard only timing baselines.
        self.timing_started=time.perf_counter();self.timing_reads=0
        self.timing_anchor[:]=False;self.max_ages[:]=0.
        self.first_update=[None for _ in self.names];self.worst_gap=[None for _ in self.names]

    def timing_details(self):
        return [dict(joint=name,interval_count=len(self.intervals[i]),
            mean_hz=None if not self.intervals[i] else
                sum(count for gap,count in self.intervals[i])/sum(gap for gap,count in self.intervals[i]),
            first_update=self.first_update[i],worst_gap=self.worst_gap[i],
            max_feedback_age_s=None if not self.timing_anchor[i] else float(self.max_ages[i]))
            for i,name in enumerate(self.names)]

    def rate_capacity(self,indices=None):
        groups=self.intervals if indices is None else [self.intervals[i] for i in indices]
        if any(len(values)<3 for values in groups):raise FeedbackUnavailable('Too few new native samples for timing')
        return min(sum(count for gap,count in values)/sum(gap for gap,count in values) for values in groups)

    def max_observed_period(self,indices=None):
        groups=self.intervals if indices is None else [self.intervals[i] for i in indices]
        if any(len(values)<3 for values in groups):raise FeedbackUnavailable('Too few new native samples for timing')
        return max(gap for values in groups for gap,count in values)

    def describe(self):
        return f'upstream get_state_snapshot: slowest={self.rate_capacity():.1f}Hz, max observed new-snapshot gap={1000*self.max_observed_period():.2f}ms'
