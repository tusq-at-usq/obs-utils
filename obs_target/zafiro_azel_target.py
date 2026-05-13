from __future__ import annotations

from pathlib import Path
import time

import astrix as at
from astrix.spatial import Rotation
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from obs_target.target import Target


class ZafiroSystemsAzElTarget(Target):
    """Target that reads ZafiroSystems AZEL CSV files.

    Expected CSV columns:
    - `Time`
    - `Altitude` (km)
    - `Azimuth` (deg)
    - `Elevation` (deg)

    Time handling:
    - By default, the CSV date is ignored and only time-of-day is used.
      The times are anchored to today's UTC date.
    - `simulate_reentry=True` replays from the first CSV point after a startup delay.
    """

    _point: at.Point

    def __init__(
        self,
        point: at.Point,
        csv_path: str,
        simulate_reentry: bool = False,
        start_delay_seconds: float = 0.0,
        use_csv_time_of_day_only: bool = True,
    ) -> None:
        self._point = point
        self._frame_ned = at.spatial.frame.ned_frame(self._point)
        self._simulate_reentry = simulate_reentry
        self._start_delay_seconds = float(start_delay_seconds)
        self._reentry_start_unix = time.time() + self._start_delay_seconds
        self._use_csv_time_of_day_only = use_csv_time_of_day_only

        df = pd.read_csv(str(Path(csv_path).expanduser()))
        required = {"Time", "Altitude", "Azimuth", "Elevation"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in ZafiroSystems AZEL CSV: {sorted(missing)}"
            )

        parsed_times = pd.to_datetime(df["Time"], utc=True)
        if self._use_csv_time_of_day_only:
            today_utc = pd.Timestamp.now("UTC").normalize()
            time_unix = parsed_times.apply(
                lambda x: (today_utc + (x - x.normalize())).timestamp()
            )
        else:
            time_unix = parsed_times.apply(lambda x: x.timestamp())

        self._time_unix = np.asarray(time_unix, dtype=float)
        self._az = np.asarray(df["Azimuth"], dtype=float)
        self._el = np.asarray(df["Elevation"], dtype=float)
        self._alt = np.asarray(df["Altitude"], dtype=float)
        self._csv_start_unix = float(self._time_unix[0])
        self._csv_end_unix = float(self._time_unix[-1])

    def _target_time(self, t_unix: float) -> float:
        if not self._simulate_reentry:
            return np.clip(t_unix, self._csv_start_unix, self._csv_end_unix)

        if t_unix <= self._reentry_start_unix:
            return self._csv_start_unix

        elapsed = t_unix - self._reentry_start_unix
        return np.clip(
            self._csv_start_unix + elapsed, self._csv_start_unix, self._csv_end_unix
        )

    def check_time_bounds(self, t_unix: float) -> tuple[bool, bool]:
        tt = self._target_time(t_unix)
        return tt >= self._csv_start_unix, tt <= self._csv_end_unix

    def get_head_pitch(self, t_unix: float) -> NDArray:
        tt = self._target_time(t_unix)
        az = np.interp(tt, self._time_unix, self._az)
        el = np.interp(tt, self._time_unix, self._el)
        return np.array([az, el])

    def get_head_pitch_rate(self, t_unix: float) -> NDArray:
        t0 = t_unix - 0.05
        t1 = t_unix + 0.05
        hp0 = self.get_head_pitch(t0)
        hp1 = self.get_head_pitch(t1)
        dt = t1 - t0
        if dt <= 0:
            return np.array([0.0, 0.0])
        return (hp1 - hp0) / dt

    def format_target_label(self, t_unix: float, hp: NDArray) -> str:
        tt = self._target_time(t_unix)
        alt = np.interp(tt, self._time_unix, self._alt)
        base = f"Az {hp[0]:>7.1f}  El {hp[1]:>6.1f} | Alt {alt:>6.1f} km"

        if not self._simulate_reentry:
            if t_unix < self._csv_start_unix:
                countdown_s = self._csv_start_unix - t_unix
                phase = f"[CSV ABSOLUTE] T-{countdown_s:>5.1f}s"
            else:
                elapsed_s = t_unix - self._csv_start_unix
                phase = f"[CSV ABSOLUTE] T+{elapsed_s:>5.1f}s"
            return f"{base} | {phase}"

        if t_unix < self._reentry_start_unix:
            countdown_s = self._reentry_start_unix - t_unix
            phase = f"[SIM REENTRY] T-{countdown_s:>5.1f}s"
        else:
            elapsed_s = t_unix - self._reentry_start_unix
            phase = f"[SIM REENTRY] T+{elapsed_s:>5.1f}s"

        return f"{base} | {phase}"

    def project_from_ned_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[NDArray, NDArray]:
        hp = self.get_head_pitch(t_unix)
        rot = Rotation.from_euler("ZYX", np.array(euler).reshape(1, -1), degrees=True)
        frame = at.Frame(rot, ref_frame=self._frame_ned)
        ray = at.Ray.from_az_el(az_el=hp, frame=frame, time=at.Time(t_unix))
        uv = ray.project_to_cam(cam)
        return uv.uv, uv.uv

    def project_from_ecef_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[NDArray, NDArray]:
        hp = self.get_head_pitch(t_unix)
        rot = Rotation.from_euler("ZYX", np.array(euler).reshape(1, -1), degrees=True)
        frame = at.Frame(rot)
        ray = at.Ray.from_az_el(az_el=hp, frame=frame, time=at.Time(t_unix))
        uv = ray.project_to_cam(cam)
        return uv.uv, uv.uv
