import os
import pathlib
import zmq
import yaml
from dataclasses import dataclass
import threading
from typing import Callable
import jax
from jax import numpy as jnp
from typing import TypeAlias, Iterable
from types import ModuleType

ArrayNS: TypeAlias = ModuleType


DEFAULT_CONFIG_PATH = os.path.join(
    pathlib.Path.home(), "obs-config/encoders_config.yaml"
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EncoderState:
    az: float  # Azimuth
    el: float  # Elevation
    az_raw: float  # Raw Azimuth
    el_raw: float  # Raw Elevation
    t: float  # PC timestamp

    @property
    def azel(self) -> list[float]:
        return [self.az, self.el]

    @property
    def azel_raw(self) -> list[float]:
        return [self.az_raw, self.el_raw]


EncoderSink = Callable[[EncoderState], None]


def encoder_extrapolator(
    last: EncoderState,
    llast: EncoderState,
    current_time: float,
) -> EncoderState:
    """Extrapolate encoder state based on last two measurements."""

    @jax.jit
    def do_extrapolation(l_azel, ll_azel, l_raw, ll_raw, dt):
        rate_azel = (l_azel - ll_azel) / dt
        rate_raw = (l_raw, ll_raw) / dt
        extrapolated_azel = l_azel + rate_azel * dt
        extrapolated_raw = l_raw + rate_raw * dt
        return extrapolated_azel, extrapolated_raw

    dt = last.t - llast.t
    if dt > 0:
        ext_azel, ext_raw = do_extrapolation(
            jnp.array(last.azel),
            jnp.array(llast.azel),
            jnp.array(last.azel_raw),
            jnp.array(llast.azel_raw),
            dt,
        )
        extrapolated_value = EncoderState(
            az=float(ext_azel[0]),
            el=float(ext_azel[1]),
            az_raw=float(ext_raw[0]),
            el_raw=float(ext_raw[1]),
            t=current_time,
        )
        return extrapolated_value
    else:
        return last


class EncoderMonitor(threading.Thread):
    """Get encoder values from the ZMQ stream"""

    config: dict
    _context: zmq.Context
    _socket: zmq.Socket
    _sinks: list[EncoderSink] | None
    _kill_event: threading.Event

    def __init__(
        self,
        config_filepath=DEFAULT_CONFIG_PATH,
        sink: EncoderSink | Iterable[EncoderSink] | None = None,
    ):
        super().__init__()
        with open(config_filepath, "r") as f:
            self.config = yaml.safe_load(f)

        if callable(sink):
            self._sinks = [sink]
        elif isinstance(sink, Iterable):
            self._sinks = list(sink)
        else:
            self._sinks = None
        self._kill_event = threading.Event()

    def __enter__(self):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.RCVTIMEO, 200)  # 100 ms timeout
        if self.config["protocol"] == "IPC":
            self._socket.connect(f"ipc://{self.config['sub_address']}")
        elif self.config["protocol"] == "TCP":
            self._socket.connect(f"tcp://*:{self.config['sub_address']}")
        else:
            raise ValueError("Unsupported protocol in config file")
        self.start()
        return self

    def __exit__(self, *exc):
        self._kill_event.set()
        self.join()

    def run(self) -> None:
        """Get the latest azimuth and elevation from the ZMQ stream."""
        try:
            while not self._kill_event.is_set():
                try:
                    message = self._socket.recv()
                    data = yaml.safe_load(message)

                    state = EncoderState(
                        az=float(data["Az"]),
                        el=float(data["El"]),
                        az_raw=float(data["Az_raw"]),
                        el_raw=float(data["El_raw"]),
                        t=float(data["Sec"]),
                    )

                    if self._sinks:
                        for sink in self._sinks:
                            sink(state)

                except zmq.Again:
                    pass

                except Exception as e:
                    print(f"Error receiving encoder data: {e}")
        finally:
            self._socket.close()
            self._context.term()
