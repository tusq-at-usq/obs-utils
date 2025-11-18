"""
1. Read trajory file in standardised format
2. Downsample trajectory
3. Create rays from current location to downsampled points
4. Project to 2D image plane

"""

import astrix as at
from astrix.spatial import Rotation
import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from functools import partial
from abc import ABC, abstractmethod


class Target(ABC):
    @abstractmethod
    def project_from_ecef_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def project_from_ned_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def check_time_bounds(self, t_unix: float) -> tuple[bool, bool]:
        pass

    @abstractmethod
    def get_head_pitch(self, t_unix: float) -> ArrayLike:
        pass


class PathTarget(Target):
    _point: at.Point
    _target_pt: at.Point
    _target_paths: at.Path
    _ray_ecef: at.Ray
    _ray_ned: at.Ray
    _frame_ned: at.Frame

    def __init__(
        self, point: at.Point, target: at.Path
    ) -> None:
        self._point = point
        self._target_path = target
        self._target_pt = target.points
        self._ray_ecef = at.Ray.from_points(self._target_pt, self._point).correct_refraction()
        self._frame_ned = at.spatial.frame.ned_frame(self._point)
        self._ray_ned = self._ray_ecef.to_frame(self._frame_ned)

    @partial(jax.jit, static_argnames=("self", "cam"))
    def _project_from_ecef_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[ArrayLike, ArrayLike]:
        """Project target points to image plane from ECEF angles.

        Args:
            euler (ArrayLike): Euler angles (yaw, pitch, roll) in radians.
        Returns:
            uv_path: 2D image coordinates of projected points.
            uv_point: 2D image coordinates of interpolated point at t_unix.
        """
        rot = Rotation.from_euler("ZYX", jnp.array(euler).reshape(1, -1), degrees=True)
        frame = at.Frame(rot, loc=self._point, backend=jnp)
        ray = self._ray_ecef.convert_to(jnp).to_frame(frame)
        uv_path = ray.project_to_cam(cam.convert_to(jnp))
        uv_point = ray.interp(
            at.Time(t_unix, backend=jnp), check_bounds=False
        ).project_to_cam(cam.convert_to(jnp))
        return uv_path.uv, uv_point.uv

    def project_from_ecef_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[NDArray, NDArray]:
        uv_path, uv_point = self._project_from_ecef_angles(euler, t_unix, cam)
        return np.array(uv_path), np.array(uv_point)

    @partial(jax.jit, static_argnames=("self", "cam"))
    def _project_from_ned_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[ArrayLike, ArrayLike]:
        """Project target points to image plane from NED angles.

        Args:
            euler (ArrayLike): Euler angles (yaw, pitch, roll) in radians.
        Returns:
            uv_path: 2D image coordinates of projected points.
            uv_point: 2D image coordinates of interpolated point at t_unix.
        """
        rot = Rotation.from_euler("ZYX", jnp.array(euler).reshape(1, -1), degrees=True)
        frame = at.Frame(rot, ref_frame=self._frame_ned, backend=jnp)
        ray = self._ray_ecef.convert_to(jnp).to_frame(frame)
        uv_path = ray.project_to_cam(cam.convert_to(jnp))
        uv_point = ray.interp(
            at.Time(t_unix, backend=jnp), check_bounds=False
        ).project_to_cam(cam.convert_to(jnp))
        return uv_path.uv, uv_point.uv

    def project_from_ned_angles(
        self, euler: ArrayLike, t_unix: float, cam: at.FixedZoomCamera
    ) -> tuple[NDArray, NDArray]:
        uv_path, uv_pt =  self._project_from_ned_angles(euler, t_unix, cam)
        return np.array(uv_path), np.array(uv_pt)

    def check_time_bounds(self, t_unix: float) -> tuple[bool, bool]:
        """Check if the given unix time is within the bounds of the target path.

        Args:
            t_unix (float): Unix time to check.
        Returns:
            start_in_bounds (bool): True if t_unix is after the start time of the target path.
            end_in_bounds (bool): True if t_unix is before the end time of the target path.
        """
        start_in_bounds = t_unix >= self._target_path.start_time.unix[0]
        end_in_bounds = t_unix <= self._target_path.end_time.unix[0]
        return start_in_bounds, end_in_bounds

    @partial(jax.jit, static_argnames=("self",))
    def _get_head_pitch(self, t_unix: float) -> ArrayLike:
        """Get the heading, pitch, and roll of the target path at the given unix time.

        Args:
            t_unix (float): Unix time to get the orientation.
        Returns:
            hpr: ArrayLike of heading, pitch, and roll in degrees.
        """
        time = at.Time(t_unix, backend=jnp)
        return self._ray_ned.convert_to(jnp).interp(time, check_bounds=False).az_el[0]

    def get_head_pitch(self, t_unix: float) -> NDArray:
        start_in_bounds, end_in_bounds = self.check_time_bounds(t_unix)
        if not start_in_bounds:
            t_unix = self._target_path.start_time.unix[0]
        elif not end_in_bounds:
            t_unix = self._target_path.end_time.unix[0]
        return np.array(self._get_head_pitch(t_unix))
