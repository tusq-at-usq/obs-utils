import os
import pathlib
import zmq
import yaml
import numpy as np
import threading
from dataclasses import dataclass
from typing import Callable
from functools import partial
from typing import TypeAlias, Iterable
from types import ModuleType
from numpy.typing import ArrayLike

from astrix.spatial import Rotation
import jax
from jax import Array
import jax.numpy as jnp

ArrayNS: TypeAlias = ModuleType


DEFAULT_CONFIG_PATH = os.path.join(
    pathlib.Path.home(), "obs-config/certus_imu_config.yaml"
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IMUState:
    heading: float
    pitch: float
    roll: float
    p: float
    q: float
    r: float
    lon: float
    lat: float
    alt: float
    t_imu: float
    t_pc: float

    @property
    def hpr(self) -> list[float]:
        return [self.heading, self.pitch, self.roll]

    @property
    def omega(self) -> list[float]:
        return [self.p, self.q, self.r]


IMUSink = Callable[[IMUState], None]


@jax.jit
def _rotation_extrapolate(hpr: Array, omega: Array, delta_t: float) -> Array:
    d_rot = Rotation.from_rotvec(omega * delta_t, degrees=True)
    rot_next = d_rot * Rotation.from_euler("ZYX", hpr, degrees=True)
    hpr_new = rot_next.as_euler("ZYX", degrees=True)
    return jnp.array(hpr_new)


def imu_extrapolate_full(
    last: IMUState, t_target: float, pc_time: bool = False
) -> IMUState:
    if pc_time:
        delta_t = t_target - last.t_pc
    else:
        delta_t = t_target - last.t_imu

    hpr_new = _rotation_extrapolate(jnp.array(last.hpr), jnp.array(last.omega), delta_t)
    return IMUState(
        heading=float(hpr_new[0]),
        pitch=float(hpr_new[1]),
        roll=float(hpr_new[2]),
        p=last.p,
        q=last.q,
        r=last.r,
        lon=last.lon,
        lat=last.lat,
        alt=last.alt,
        t_imu=last.t_imu + delta_t,
        t_pc=last.t_pc + delta_t,
    )


def imu_extrapolate_simple(
    last: IMUState, t_target: float, pc_time: bool = False
) -> IMUState:
    if pc_time:
        delta_t = t_target - last.t_pc
    else:
        delta_t = t_target - last.t_imu
    hpr_new = np.array(last.hpr) + np.array(last.omega) * delta_t
    return IMUState(
        heading=float(hpr_new[0]),
        pitch=float(hpr_new[1]),
        roll=float(hpr_new[2]),
        p=last.p,
        q=last.q,
        r=last.r,
        lon=last.lon,
        lat=last.lat,
        alt=last.alt,
        t_imu=last.t_imu + delta_t,
        t_pc=last.t_pc + delta_t,
    )


class CertusMonitor(threading.Thread):
    """Get IMU values from the ZMQ stream"""

    _config: dict
    _context: zmq.Context
    _socket: zmq.Socket
    rot_fixed: Rotation
    _euler_offset: tuple[float, float, float]
    _kill_event: threading.Event
    _sinks: list[IMUSink] | None

    def __init__(
        self,
        config_filepath=DEFAULT_CONFIG_PATH,
        sink: IMUSink | Iterable[IMUSink] | None = None,
    ):
        super().__init__()
        with open(config_filepath, "r") as f:
            self._config = yaml.safe_load(f)
        self._euler_offset = self._config.get("euler_offset", [0,0,0])
        self.rot_fixed = Rotation.from_euler(
            "ZYX", jnp.array(self._euler_offset), degrees=True
        )
        self._kill_event = threading.Event()
        if callable(sink):
            self._sinks = [sink]
        elif isinstance(sink, Iterable):
            self._sinks = list(sink)
        else:
            self._sinks = None

    def __enter__(self):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        if self._config["protocol"] == "IPC":
            self._socket.connect(f"ipc://{self._config['address']}")
        elif self._config["protocol"] == "TCP":
            self._socket.connect(f"tcp://{self._config['address']}")
        else:
            raise ValueError("Unsupported protocol in config file")
        self.start()
        return self

    def __exit__(self, *exc):
        self._kill_event.set()
        self.join()

    @partial(jax.jit, static_argnames=("self",))
    def rotate_rot(self, hpr: ArrayLike) -> ArrayLike:
        """Rotate HPR values based on a fixed azimuth rotation."""
        rot_hpr = Rotation.from_euler("ZYX", jnp.array(hpr), degrees=True)
        rot_rotated = rot_hpr * self.rot_fixed
        return rot_rotated.as_euler("ZYX", degrees=True)

    @partial(jax.jit, static_argnames=("self",))
    def rotate_omega(self, omega: ArrayLike) -> ArrayLike:
        """Rotate angular velocity values based on a fixed azimuth rotation."""
        omega_rotated = self.rot_fixed.apply(jnp.array(omega))
        return omega_rotated

    def run(self) -> None:
        try:
            while not self._kill_event.is_set():
                try:
                    message = self._socket.recv()
                    data = yaml.safe_load(message)

                    euler = [data["Head"], data["Pitch"], data["Roll"]]
                    omega = [data["w_Roll"], data["w_Pitch"], data["w_Head"]]
                    if self._euler_offset:
                        euler = self.rotate_rot(euler)
                        omega = self.rotate_omega(omega)

                    state = IMUState(
                        heading=float(euler[0]),
                        pitch=float(euler[1]),
                        roll=float(euler[2]),
                        p=float(omega[0]),
                        q=float(omega[1]),
                        r=float(omega[2]),
                        lon=data["Lon"],
                        lat=data["Lat"],
                        alt=data["Alt"],
                        t_imu=data["Sec"],
                        t_pc=data["PC_Time"],
                    )

                    t_now = float(data["PC_Time"])

                    if self._sinks:
                        for sink in self._sinks:
                            sink(state)

                except zmq.Again:
                    # No new message, use the last known values
                    pass
        finally:
            self._socket.close()
            self._context.term()
