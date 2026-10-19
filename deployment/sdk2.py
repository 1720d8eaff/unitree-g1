"""G1 29DoF arm_sdk transport with bounded commands and a separate watchdog.

No Dex3, waist, locomotion or balancing controller is implemented here.
The real transport is lazy and requires explicit hardware enablement.
"""
import threading
import time
import numpy as np
from .policy import JOINT_IDS, finite


class ArmController:
    def __init__(self, transport, *, lower, upper, max_speed, kp, kd,
                 state_timeout_s=.1, command_timeout_s=.25, clock=time.monotonic):
        self.transport, self.clock = transport, clock
        self.lower, self.upper = finite(lower, (14,), "lower"), finite(upper, (14,), "upper")
        self.speed = finite(max_speed, (14,), "max_speed")
        self.kp, self.kd = finite(kp, (14,), "kp"), finite(kd, (14,), "kd")
        if np.any(self.lower >= self.upper) or np.any(self.speed <= 0) or np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("Invalid joint limits, speed or gains")
        if not np.isfinite([state_timeout_s, command_timeout_s]).all() or min(state_timeout_s, command_timeout_s) <= 0:
            raise ValueError("Watchdog timeouts must be positive")
        self.state_timeout, self.command_timeout = state_timeout_s, command_timeout_s
        self.target = self.last = None
        self.target_stamp = self.last_tick = None
        self.lock = threading.Lock()
        self.stopped = False

    def submit(self, target, observation_timestamp, *, source, observation_source):
        now = self.clock()
        q = finite(target, (14,), "target")
        if (not np.isfinite(observation_timestamp) or not 0 <= now - observation_timestamp <= self.command_timeout):
            raise ValueError("Refusing future or stale observation")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("Target outside declared robot limits")
        if observation_source not in {"synthetic", "physics_simulation", "real_robot"}:
            raise ValueError("Unknown observation provenance")
        if not self.transport.synthetic and (source == "synthetic" or observation_source != "real_robot"):
            raise ValueError("Real transport needs model commands derived from real observations")
        if source not in {"synthetic", "gr00t_local", "gr00t_remote"}:
            raise ValueError("Unknown action provenance")
        with self.lock:
            if self.stopped:
                raise RuntimeError("Controller has stopped")
            if self.target_stamp is not None and observation_timestamp < self.target_stamp:
                raise ValueError("Out-of-order command")
            self.target, self.target_stamp = q, observation_timestamp

    def tick(self):
        now = self.clock()
        state = self.transport.read()
        with self.lock:
            dt = 0. if self.last_tick is None else now - self.last_tick
            if not np.isfinite(now) or dt < 0:
                self.transport.release()
                raise ValueError("Monotonic clock moved backwards")
            self.last_tick = now
            if self.stopped or state is None:
                self.transport.release()
                return "released"
            stamp, measured = state
            measured = finite(measured, (14,), "robot state")
            if (not np.isfinite(stamp) or not 0 <= now - stamp <= self.state_timeout
                or self.target_stamp is None or not 0 <= now - self.target_stamp <= self.command_timeout
                or np.any(measured < self.lower) or np.any(measured > self.upper)):
                self.last = None
                self.transport.release()
                return "watchdog_release"
            if self.last is None:
                self.last = measured.copy()
            # A delayed control tick never permits a large catch-up jump.
            step = self.speed * min(dt, .02)
            self.last += np.clip(self.target - self.last, -step, step)
            self.transport.write(self.last.copy(), self.kp, self.kd)
            return "active"

    def close(self):
        with self.lock:
            self.stopped = True
            self.transport.release()


class MemoryTransport:
    synthetic = True

    def __init__(self, joints, clock=time.monotonic):
        self.q, self.clock, self.commands = finite(joints, (14,), "joints"), clock, []
        self.weight = 0.

    def read(self):
        return self.clock(), self.q.copy()

    def write(self, joints, kp, kd):
        self.q, self.weight = joints.copy(), 1.
        self.commands.append(dict(q=joints.tolist(), weight=1., source="synthetic"))

    def release(self):
        self.weight = 0.


class UnitreeTransport:
    synthetic = False

    def __init__(self, interface, *, enable_hardware=False, robot_dof=29, clock=time.monotonic):
        if not enable_hardware or robot_dof != 29 or not interface:
            raise ValueError("Explicitly enable a calibrated G1 29DoF network interface")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        self.clock, self.state, self.lock = clock, None, threading.Lock()
        self.command = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        ChannelFactoryInitialize(0, interface)
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._state, 10)

    def _state(self, message):
        q = np.array([message.motor_state[i].q for i in JOINT_IDS], dtype=float)
        if np.isfinite(q).all():
            with self.lock:
                self.state = self.clock(), q

    def read(self):
        with self.lock:
            return None if self.state is None else (self.state[0], self.state[1].copy())

    def write(self, joints, kp, kd):
        for index, q, p, d in zip(JOINT_IDS, joints, kp, kd):
            cmd = self.command.motor_cmd[index]
            cmd.q, cmd.dq, cmd.tau, cmd.kp, cmd.kd = float(q), 0., 0., float(p), float(d)
        self.command.motor_cmd[29].q = 1.
        self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)

    def release(self):
        self.command.motor_cmd[29].q = 0.
        self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)


class WatchdogLoop:
    """Controller lifetime is independent of potentially blocked model inference."""
    def __init__(self, controller, period_s=.02):
        if not np.isfinite(period_s) or not 0 < period_s <= .02:
            raise ValueError("Watchdog loop period must be at most 20ms")
        self.controller, self.period = controller, period_s
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="g1-arm-watchdog", daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                self.controller.tick()
                self.stop.wait(self.period)
        except Exception as exc:
            self.error = exc
        finally:
            self.controller.close()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1.)
        self.controller.close()
