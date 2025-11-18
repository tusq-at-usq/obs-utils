from __future__ import annotations
import threading
import zmq
import time
import warnings
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, replace
from typing import Callable
from queue import SimpleQueue, Empty

from obs_target.target import Target
from obs_certus.monitor import IMUState


@dataclass
class GimbalState:
    az: float = 0.0
    el: float = 0.0
    t_pos: float = 0.0
    sp_az: float = 0.0
    sp_el: float = 0.0
    u_az: float = 0.0
    u_el: float = 0.0
    int_err_az: float = 0.0
    int_err_el: float = 0.0
    t_ctrl_update: float = 0.0
    t_cmd_update: float = 0.0
    az_limit_min: float = -170.0
    az_limit_max: float = 170.0
    el_limit_min: float = -70.0
    el_limit_max: float = 89.9
    mode: str = "manual"

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.az, self.el])

    @property
    def u(self) -> np.ndarray:
        return np.array([self.u_az, self.u_el])

    @property
    def sp(self) -> np.ndarray:
        return np.array([self.sp_az, self.sp_el])

    @property
    def int_err(self) -> np.ndarray:
        return np.array([self.int_err_az, self.int_err_el])


class SharedGimbalState:
    _state: GimbalState
    _lock: threading.RLock

    def __init__(self) -> None:
        self._state = GimbalState(
            az=0.0,
            el=0.0,
            sp_az=0.0,
            sp_el=0.0,
            u_az=0.0,
            u_el=0.0,
            int_err_az=0.0,
            int_err_el=0.0,
            t_pos=time.time(),
            t_ctrl_update=time.time(),
            t_cmd_update=time.time(),
        )
        self._lock = threading.RLock()

    def update_from_recv(self, new_state: GimbalState) -> None:
        with self._lock:
            s = self._state
            s.az = new_state.az
            s.el = new_state.el
            s.t_pos = new_state.t_pos

    def update_from_send(self, new_state: GimbalState) -> None:
        with self._lock:
            s = self._state
            s.u_az = new_state.u_az
            s.u_el = new_state.u_el
            s.mode = new_state.mode
            s.t_cmd_update = new_state.t_cmd_update

    def update_from_ctrl(self, new_state: GimbalState) -> None:
        with self._lock:
            s = self._state
            s.sp_az = new_state.sp_az
            s.sp_el = new_state.sp_el
            s.int_err_az = new_state.int_err_az
            s.int_err_el = new_state.int_err_el
            s.t_ctrl_update = new_state.t_ctrl_update

    def update_limit(self, axis: str, end: str) -> None:
        with self._lock:
            if axis == "az":
                if end == "min":
                    self._state.az_limit_min = self._state.az
                elif end == "max":
                    self._state.az_limit_max = self._state.az
            elif axis == "el":
                if end == "min":
                    self._state.el_limit_min = self._state.el
                elif end == "max":
                    self._state.el_limit_max = self._state.el

    def update_single_field(self, field_name: str, value: float | str) -> None:
        with self._lock:
            if hasattr(self._state, field_name):
                setattr(self._state, field_name, value)
            else:
                warnings.warn(f"GimbalState has no field named '{field_name}'")

    def snapshot(self) -> GimbalState:
        with self._lock:
            return replace(self._state)

    # _lock: threading.RLock = field(default_factory=threading.RLock)


shared_gimbal_state = SharedGimbalState()

GimbalSink = Callable[[GimbalState], None]


class GimbalRecv(threading.Thread):
    _kill_switch: threading.Event
    _context: zmq.Context | None
    _own_context: bool
    _pos_socket: zmq.Socket
    _state: SharedGimbalState
    _sink: GimbalSink | None

    def __init__(
        self,
        state: SharedGimbalState,
        sink: GimbalSink | None = None,
    ):
        super().__init__()
        self._kill_switch = threading.Event()
        self._sink = sink
        self._state = state

    def connect(self, ctx: zmq.Context | None = None) -> None:
        if ctx is None:
            self._context = zmq.Context()
            self._own_context = True
        else:
            self._context = ctx
            self._own_context = False
        self._pos_socket = self._context.socket(zmq.SUB)
        self._pos_socket.setsockopt(zmq.RCVTIMEO, 200)  # 500ms timeout on reply
        self._pos_socket.setsockopt(zmq.CONFLATE, 1)  # Only want most recent command
        self._pos_socket.subscribe("P")
        self._pos_socket.connect("tcp://scoti.local:60000")

    def run(self):
        try:
            while not self._kill_switch.is_set():
                try:
                    recv = self._pos_socket.recv_string()
                    recv = recv.split(",")
                    p_az = float(recv[1][1:])  # Remove 'A' prefix
                    p_el = float(recv[2][1:])  # Remove 'E' prefix
                    t_data = float(recv[3][1:])  # Remove 'T' prefix

                    self._state.update_from_recv(
                        GimbalState(
                            az=p_az,
                            el=p_el,
                            t_pos=t_data,
                        )
                    )

                    if self._sink is not None:
                        self._sink(self._state.snapshot())

                except zmq.Again:
                    warnings.warn(
                        "Timeout waiting for position data from SCOTI controller"
                    )
                except Exception as e:
                    print("Error receiving position:", e)

        finally:
            self._pos_socket.close()
            if self._own_context and self._context is not None:
                self._context.term()
            print("Controller stopped gracefully")

    def stop(self):
        """Stop the controller thread."""
        self._kill_switch.set()


class GimbalSend(threading.Thread):
    _kill_switch: threading.Event
    _new_msg_event: threading.Event

    _context: zmq.Context
    _own_context: bool
    _speed_set_socket: zmq.Socket

    _state: SharedGimbalState
    _msg_queue: SimpleQueue[str]
    _mode: str

    def __init__(self, state: SharedGimbalState, ctx: zmq.Context | None = None):
        super().__init__()
        self._kill_switch = threading.Event()

        self._state = state
        self._mode = "manual"
        self._msg_queue = SimpleQueue()
        self._new_msg_event = threading.Event()

    def connect(self, ctx: zmq.Context | None = None) -> None:
        if ctx is None:
            self._context = zmq.Context()
            self._own_context = True
        else:
            self._context = ctx
            self._own_context = False
        self._speed_set_socket = self._context.socket(zmq.REQ)
        self._speed_set_socket.setsockopt(zmq.RCVTIMEO, 200)  # 200ms timeout on reply
        self._speed_set_socket.connect("tcp://scoti.local:60004")
        self.send_speed(0.0, 0.0, force=True)

    def set_limit(self, axis: str, end: str) -> None:
        """
        Thread-safe method to set the limit to current position for a given axis.
        Setting to position prevents current point being outside limits

        Args:
            axis: Axis to set the limit for ('az' or 'el')
            end: End of the axis to set the limit for ('min' or 'max')
        """
        message = None
        if axis == "az":
            if end == "min":
                message = "L,LA"
                self._state.update_limit("az", "min")
            elif end == "max":
                message = "L,UA"
                self._state.update_limit("az", "max")
        elif axis == "el":
            if end == "min":
                message = "L,LE"
                self._state.update_limit("el", "min")
            elif end == "max":
                message = "L,UE"
                self._state.update_limit("el", "max")
        if message is not None:
            self._msg_queue.put(message)
            self._new_msg_event.set()
        else:
            warnings.warn("Invalid axis or end specified for set_limit")

    def reset_limits(self):
        """Resets limits to:
        - Azimuth: +170 to -170 degrees
        - Elevation: +89.9 to -70 degrees
        """
        self._msg_queue.put("L,R")
        self._state.update_single_field("az_limit_min", -170.0)
        self._state.update_single_field("az_limit_max", 170.0)
        self._state.update_single_field("el_limit_min", -70.0)
        self._state.update_single_field("el_limit_max", 89.9)

    def set_home(self, axis: str) -> None:
        """Sets the current position as home position"""
        message = None
        if axis == "az":
            message = "H,A"
        elif axis == "el":
            message = "H,E"
        if message is not None:
            self._msg_queue.put(message)
            self._new_msg_event.set()
        else:
            warnings.warn("Invalid axis specified for set_home")

    @property
    def mode(self) -> str:
        """
        Get the current control mode.

        Returns:
            Current control mode as a string
        """
        return self._mode

    def set_mode(self, mode: str) -> None:
        """
        Thread-safe method to set the control mode.

        Args:
            mode: One of 'manual' or 'tracking'
        """
        if mode in ["manual", "tracking"]:
            if mode != "tracking":
                self.send_speed(0.0, 0.0, force=True)
                self._new_msg_event.set()
            self._mode = mode
            self._state.update_single_field("mode", mode)
            self._new_msg_event.set()

    def send_speed(self, az_speed: float, el_speed: float, force=False) -> None:
        """
        Thread-safe method to send speed command in manual mode.

        Args:
            az_speed: Azimuth speed in degrees per second
            el_speed: Elevation speed in degrees per second
        """
        # Safety check that in manual mode
        if self.mode == "manual" and not force:
            warnings.warn("Cannot send speed command in non-manual mode")
        else:
            self._msg_queue.put(f"S,A{az_speed:.3f},E{el_speed:.3f}")
            self._new_msg_event.set()
            self._state.update_from_send(
                GimbalState(
                    u_az=az_speed,
                    u_el=el_speed,
                    mode=self.mode,
                    t_cmd_update=time.time(),
                )
            )

    def _send_command(self, command: str) -> bool:
        """
        Send a command to the speed set socket.
        Should only be called from the controller thread.

        Args:
            command: Command string to send
        """
        try:
            self._speed_set_socket.send_string(command)
            ret_msg = self._speed_set_socket.recv_string()
            if ret_msg != "OK":
                print("Error sending command to speed set socket")
            else:
                return True
        except zmq.Again:
            warnings.warn("Timeout waiting for acknowledgment SCOTI controller")
        except zmq.ContextTerminated:
            warnings.warn("ZMQ context terminated")
        except zmq.ZMQError as e:
            warnings.warn(f"ZMQ Error sending command to SCOTI controller: {e}")
            pass
        return False

    def run(self):
        self.send_speed(0.0, 0.0, force=True)
        try:
            while not self._kill_switch.is_set():
                self._new_msg_event.wait(timeout=0.1)
                if self._kill_switch.is_set():
                    break
                self._new_msg_event.clear()
                while True:
                    try:
                        msg = self._msg_queue.get_nowait()
                    except Empty:
                        break
                    self._send_command(msg)
        finally:
            self._speed_set_socket.close()
            if self._own_context and self._context is not None:
                self._context.term()
            print("GimbalSend stopped gracefully")

    def stop(self):
        """Stop the controller thread."""
        self.send_speed(0.0, 0.0)
        self._new_msg_event.set()
        self._kill_switch.set()


ControlSink = Callable[[float, float], None]


class GimbPIController(threading.Thread):
    Kp: float
    Ki: float
    Kff: float
    integral_error: NDArray
    feed_forward: NDArray
    _target: Target
    _gimbal_state: SharedGimbalState
    _ctrl_sink: ControlSink | None
    _prev_t: float
    _imu_state: IMUState
    _pc_time: bool
    _update_event: threading.Event
    _kill_switch: threading.Event
    _lock: threading.RLock
    _u_lpf_t: float
    _u: NDArray

    def __init__(
        self,
        target: Target,
        gimbal_state: SharedGimbalState,
        ctrl_sink: ControlSink | None = None,
        _pc_time: bool = False,
    ):
        super().__init__()
        self.Kp = 1.0
        self.Ki = 0.0
        self.Kff = 1
        self.int_limit = 200.0
        self._u_lpf_t = 0.0
        self._u = np.zeros(2)
        self.deadband = 0.0
        self.integral_error = np.zeros(2)
        self.feed_forward = np.zeros(2)
        self._target = target
        self._gimbal_state = gimbal_state
        self._prev_t = time.time()
        self._ctrl_sink = ctrl_sink
        self._update_event = threading.Event()
        self._pc_time = _pc_time
        self._kill_switch = threading.Event()
        self._lock = threading.RLock()

    def new_data(self, imu_state: IMUState) -> None:
        with self._lock:
            self._imu_state = imu_state
            self._update_event.set()

    def pc_time(self, value: bool) -> None:
        with self._lock:
            self._pc_time = value

    @staticmethod
    def angle_diff_deg(a: float | NDArray, b: float | NDArray) -> float | NDArray:
        d = (a - b + 180) % 360 - 180
        return d

    def get_t(self) -> float:
        with self._lock:
            if self._pc_time:
                return self._imu_state.t_pc
            else:
                return self._imu_state.t_imu

    def reset_integral(self) -> None:
        with self._lock:
            self.integral_error = np.zeros(2)
            self._prev_t = self.get_t()

    def update_control(self) -> dict[str, NDArray | list[float]]:
        with self._lock:
            hp_imu = np.array(self._imu_state.hpr[0:2])
            t = self.get_t()
        hp_target = np.array(self._target.get_head_pitch(t))
        error = np.array(self.angle_diff_deg(np.array(hp_target), hp_imu))
        error[np.abs(error) < self.deadband] = 0.0
        dt = t - self._prev_t
        self.integral_error += error * dt
        self.integral_error = np.clip(
            self.integral_error, -self.int_limit, self.int_limit
        )
        self.feed_forward = np.array(self._target.get_head_pitch_rate(t))

        u_ = self.Kp * error + self.Ki * self.integral_error + self.Kff *  self.feed_forward
        self._u = (1 - self._u_lpf_t) * u_ + self._u_lpf_t * self._u
        control_output = self._u
        self._prev_t = t
        return {
            "ctrl": control_output,
            "sp": hp_target,
            "state": hp_imu,
            "error": error,
        }

    def run(self):
        while not self._kill_switch.is_set():
            if self._update_event.wait(timeout=0.1):
                if self._kill_switch.is_set():
                    break
                self._update_event.clear()
                if self._gimbal_state.snapshot().mode == "tracking":
                    data = self.update_control()
                    if self._ctrl_sink is not None:
                        self._ctrl_sink(data["ctrl"][0], data["ctrl"][1])
                    self._gimbal_state.update_from_ctrl(
                        GimbalState(
                            sp_az=data["sp"][0],
                            sp_el=data["sp"][1],
                            int_err_az=self.integral_error[0],
                            int_err_el=self.integral_error[1],
                            t_ctrl_update=time.time(),
                        )
                    )

    def stop(self):
        """Stop the controller thread."""
        self._update_event.set()
        self._kill_switch.set()


class GimbalController:
    _gimbal_state: SharedGimbalState
    _gimbal_recv_thread: GimbalRecv
    _gimbal_send_thread: GimbalSend
    _gimbal_pi_thread: GimbPIController
    _ctx: zmq.Context

    @property
    def recv_thread(self) -> GimbalRecv:
        return self._gimbal_recv_thread

    @property
    def send_thread(self) -> GimbalSend:
        return self._gimbal_send_thread

    @property
    def pi_thread(self) -> GimbPIController:
        return self._gimbal_pi_thread

    def set_imu_state(self, imu_state: IMUState) -> None:
        self._gimbal_pi_thread.new_data(imu_state)

    def set_mode(self, mode: str) -> None:
        self._gimbal_pi_thread.reset_integral()
        self._gimbal_send_thread.set_mode(mode)

    def set_home(self, axis: str) -> None:
        self._gimbal_send_thread.set_home(axis)

    def set_limit(self, axis: str, end: str) -> None:
        self._gimbal_send_thread.set_limit(axis, end)

    def reset_limits(self) -> None:
        self._gimbal_send_thread.reset_limits()

    def __init__(self, target: Target, sink: GimbalSink | None = None):
        self._gimbal_state = shared_gimbal_state
        self._gimbal_recv_thread = GimbalRecv(state=self._gimbal_state, sink=sink)
        self._gimbal_send_thread = GimbalSend(state=self._gimbal_state)
        self._gimbal_pi_thread = GimbPIController(
            target=target,
            gimbal_state=self._gimbal_state,
            ctrl_sink=self._gimbal_send_thread.send_speed,
        )

    def stop(self):
        self._gimbal_recv_thread.stop()
        self._gimbal_send_thread.stop()
        self._gimbal_pi_thread.stop()

    def __enter__(self):
        self._ctx = zmq.Context()
        self._gimbal_recv_thread.connect(self._ctx)
        self._gimbal_send_thread.connect(self._ctx)
        self._gimbal_recv_thread.start()
        self._gimbal_send_thread.start()
        self._gimbal_pi_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self._gimbal_recv_thread.join()
        self._gimbal_send_thread.join()
        self._gimbal_pi_thread.join()
        self._ctx.term()
