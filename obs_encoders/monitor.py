import os
import pathlib
import zmq
import yaml
from collections import deque
from statistics import median
from dataclasses import dataclass
import threading
from typing import Callable
import jax
from jax import numpy as jnp
from typing import TypeAlias, Iterable
from types import ModuleType

ArrayNS: TypeAlias = ModuleType


DEFAULT_CONFIG_PATH = os.path.join(
    pathlib.Path(__file__).resolve().parent, "encoder_config.yaml"
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
    _swap_az_el: bool

    # Median filter applied to az/el before values reach sinks (overlay/prediction
    # consumers only). Raw values and the CSV log written by the broadcaster are
    # never touched by this filter.
    #
    # _FILTER_WINDOW: number of recent samples kept. Must be odd for a clean median.
    # _SPIKE_THRESHOLD_DEG: a new sample whose distance from the current median
    #   exceeds this value (degrees) is discarded outright and never enters the
    #   buffer, preventing spikes from poisoning future medians.
    #   Overridden by config key `max_rate_deg_per_sec` (physics-based gate takes
    #   precedence when a previous timestamp is available).
    _FILTER_WINDOW = 7
    _SPIKE_THRESHOLD_DEG = 5.0

    def __init__(
        self,
        config_filepath=DEFAULT_CONFIG_PATH,
        sink: EncoderSink | Iterable[EncoderSink] | None = None,
        swap_az_el: bool | None = None,
    ):
        super().__init__()
        with open(config_filepath, "r") as f:
            self.config = yaml.safe_load(f)

        if swap_az_el is None:
            self._swap_az_el = bool(self.config.get("swap_az_el", False))
        else:
            self._swap_az_el = swap_az_el

        self._max_rate = float(self.config.get("max_rate_deg_per_sec", 60.0))

        if callable(sink):
            self._sinks = [sink]
        elif isinstance(sink, Iterable):
            self._sinks = list(sink)
        else:
            self._sinks = None
        self._kill_event = threading.Event()
        self._az_buf = deque(maxlen=self._FILTER_WINDOW)
        self._el_buf = deque(maxlen=self._FILTER_WINDOW)
        self._last_accepted: EncoderState | None = None

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

    def _apply_axis_config(self, state: EncoderState) -> EncoderState:
        if not self._swap_az_el:
            return state
        return EncoderState(
            az=state.el,
            el=state.az,
            az_raw=state.el_raw,
            el_raw=state.az_raw,
            t=state.t,
        )

    def _filter_state(self, state: EncoderState) -> EncoderState:
        """Reject physically impossible samples, then median-smooth for sinks.

        Raw counts are passed through unchanged — only the sink-facing az/el are
        smoothed here. The CSV log written by the broadcaster in run.py is produced
        upstream of this filter and is unaffected.

        Two-stage rejection:
        1. Rate gate (physics): if a previous accepted sample exists, compute the
           implied angular rate for az and el separately.  Any axis whose rate
           exceeds `max_rate_deg_per_sec` (config) flags the whole sample as a
           spike and the sample is dropped without touching the buffers.
        2. Spike gate (fallback): when the buffer is warm but no rate can be
           computed (first sample, or identical timestamps), reject values that
           deviate from the current median by more than _SPIKE_THRESHOLD_DEG.
        3. Median smoothing: output is the median of the accepted window, giving
           additional robustness against any outliers that slip through.
        """
        last = self._last_accepted

        if last is not None and (state.t - last.t) > 0:
            dt = state.t - last.t
            az_rate = abs(state.az - last.az) / dt
            el_rate = abs(state.el - last.el) / dt
            if az_rate > self._max_rate or el_rate > self._max_rate:
                # Physically impossible — drop the sample entirely
                az_out = median(self._az_buf) if self._az_buf else last.az
                el_out = median(self._el_buf) if self._el_buf else last.el
                return EncoderState(az=az_out, el=el_out,
                                    az_raw=state.az_raw, el_raw=state.el_raw,
                                    t=state.t)
        else:
            # Fallback spike gate: deviation from current median
            def _over_threshold(value: float, buf: deque) -> bool:
                return len(buf) > 0 and abs(value - median(buf)) > self._SPIKE_THRESHOLD_DEG

            if _over_threshold(state.az, self._az_buf) or _over_threshold(state.el, self._el_buf):
                az_out = median(self._az_buf) if self._az_buf else state.az
                el_out = median(self._el_buf) if self._el_buf else state.el
                return EncoderState(az=az_out, el=el_out,
                                    az_raw=state.az_raw, el_raw=state.el_raw,
                                    t=state.t)

        self._az_buf.append(state.az)
        self._el_buf.append(state.el)
        self._last_accepted = state
        return EncoderState(
            az=median(self._az_buf),
            el=median(self._el_buf),
            az_raw=state.az_raw,
            el_raw=state.el_raw,
            t=state.t,
        )

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
                    state = self._apply_axis_config(state)
                    state = self._filter_state(state)

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
