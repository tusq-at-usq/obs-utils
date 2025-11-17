# pyright: standard

"""
Interface class definitions
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import threading
from typing import Tuple, Union
from abc import ABC, abstractmethod
import pathlib
import os
import datetime
import time
import queue
from pathlib import Path
import csv

from astrix import FixedZoomCamera


@dataclass(frozen=True)
class Frame:
    pixels: NDArray
    gain: float
    exposure: float
    timestamp: float  # POSIX timestamp in seconds
    cam_name: str = "unnamed_camera"

    def copy(self) -> Frame:
        """
        Create a deep copy of the frame and its metadata.

        Returns:
            Frame: A new Frame object with copied data
        """
        return Frame(
            pixels=self.pixels.copy(),  # Deep copy of numpy array
            gain=self.gain,
            exposure=self.exposure,
            timestamp=self.timestamp,
            cam_name=self.cam_name,
        )


class CameraInterface(ABC):
    # Class attributes that must be defined by derived classes

    FRAME_RES: Tuple[int, int]
    SENSOR_SIZE: Tuple[float, float]
    DTYPE: str = "uint8"  # Data type of the image data
    _gain_set: float | str = 0.0
    _exposure_set: float = 0.0
    _bandwidth_val: Union[float, str]
    _last_frame_time: float
    _actual_frame_rate: float
    _frame_count: int
    _frame_rate_window: int
    _settings_lock: threading.RLock

    def __init__(self) -> None:
        """Initialize the camera with thread safety."""

        self._settings_lock = threading.RLock()
        self._last_frame_time = 0.0
        self._actual_frame_rate = 0.0
        self._frame_count = 0
        self._frame_rate_window = 20

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __enter__(self) -> CameraInterface:
        """Enter the runtime context related to this object."""
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the runtime context related to this object."""
        pass

    # Abstract methods that must be implemented by derived classes
    @abstractmethod
    def start_video(self) -> None:
        """
        Start video capture mode.
        Must be implemented by derived classes.
        """
        pass

    @abstractmethod
    def _get_frame(self) -> Frame:
        """
        Hardware-specific implementation of frame capture.
        Must be implemented by derived classes.

        Returns:
            Frame: A Frame object containing the image data and metadata
            timestamp: Timestamp when the frame was captured
        """
        pass

    @abstractmethod
    def _get_gain(self) -> float:
        """
        Hardware-specific implementation of gain retrieval.
        Must be implemented by derived classes.

        Returns:
            float: Current gain value
        """
        pass

    @abstractmethod
    def _get_exposure(self) -> float:
        """
        Hardware-specific implementation of exposure retrieval.
        Must be implemented by derived classes.

        Returns:
            float: Current exposure time in microseconds
        """
        pass

    @abstractmethod
    def _set_exposure(self, exp: float) -> None:
        """
        Hardware-specific implementation of exposure setting.
        Must be implemented by derived classes.

        Args:
            exp: Exposure time in microseconds
        """
        pass

    @abstractmethod
    def _set_gain(self, gain: Union[float, str]) -> None:
        """
        Hardware-specific implementation of gain setting.
        Must be implemented by derived classes.

        Args:
            gain: Gain value or "auto"
        """
        pass

    # Optional camera features
    def _set_frame_rate(self, fps: float) -> None:
        """
        Hardware-specific implementation of frame rate setting.
        Must be implemented by derived classes that support frame rate control.

        Args:
            fps: Frames per second
        """
        raise NotImplementedError("Frame rate control not supported by this camera")

    def _set_bandwidth(self, bw: Union[float, int, str]) -> None:
        """
        Hardware-specific implementation of bandwidth setting.
        Must be implemented by derived classes that support bandwidth control.

        Args:
            bw: Bandwidth value in percentage or "auto"/"min"/"max"
        """
        raise NotImplementedError("Bandwidth control not supported by this camera")

    def convert_for_monitoring(self, frame: Frame) -> Frame:
        """
        Convert the frame for monitoring purposes.
        This method can be overridden by derived classes if needed.

        Args:
            frame (Frame): The frame to convert.

        Returns:
            Frame: The converted frame.
        """
        return frame

    # Additional methods

    # def read_config(self, filepath) -> None:
    #     """
    #     Read camera settings from a YAML configuration file.
    #
    #     Args:
    #         filepath (str): Path to the YAML configuration file
    #     """
    #     with open(filepath, "r") as file:
    #         config = yaml.safe_load(file)
    #
    #     with self._settings_lock:
    #         if "name" in config:
    #             self.name = config["name"]
    #         if "exposure" in config:
    #             self._exposure_set = config["exposure"]
    #             self._set_exposure(self._exposure_set)
    #         if "gain" in config:
    #             self._gain_set = config["gain"]
    #             self._set_gain(self._gain_set)
    #         if "bandwidth" in config:
    #             self._bandwidth_val = config["bandwidth"]
    #             self._set_bandwidth(self._bandwidth_val)
    #         if "frame_rate" in config:
    #             self._set_frame_rate(config["frame_rate"])

    def get_frame(self) -> Frame:
        """
        Get a single frame from the camera and update frame rate statistics.

        Returns:
            Frame: A Frame object containing the image data and metadata
        """
        frame = self._get_frame()

        if self._last_frame_time is not None:
            # Calculate frame interval
            frame_interval = frame.timestamp - self._last_frame_time
            if frame_interval > 0:
                # Update actual frame rate using a moving average
                self._frame_count += 1
                self._actual_frame_rate = (
                    self._actual_frame_rate
                    * (self._frame_rate_window - 1)
                    / self._frame_rate_window
                    + (1 / frame_interval) / self._frame_rate_window
                )
        self._last_frame_time = frame.timestamp
        return frame

    def set_exposure(self, exp: float) -> None:
        """
        Thread-safe method to set the camera exposure time.

        Args:
            exp: Exposure time in microseconds
        """
        with self._settings_lock:
            self._exposure_set = exp
            self._set_exposure(exp)

    def set_gain(self, gain: Union[float, str]) -> None:
        """
        Thread-safe method to set the camera gain value.

        Args:
            gain: Gain value or "auto"
        """
        with self._settings_lock:
            self._gain_set = gain
            self._set_gain(gain)

    def set_bandwidth(self, bw: Union[float, int, str] = 80) -> None:
        """
        Thread-safe method to set the camera bandwidth.

        Args:
            bw: Bandwidth value in percentage or "auto"/"min"/"max"
        """
        with self._settings_lock:
            self._bandwidth_val = bw
            self._set_bandwidth(bw)

    @property
    def exposure_set(self) -> float:
        """
        Thread-safe getter for current exposure value.

        Returns:
            float: Current exposure value in microseconds
        """
        with self._settings_lock:
            return self._exposure_set

    @property
    def exposure(self) -> float:
        """
        Thread-safe getter for current exposure value.

        Returns:
            float: Current exposure value in microseconds
        """
        with self._settings_lock:
            return self._get_exposure()

    @property
    def gain(self) -> float:
        """
        Thread-safe getter for current gain value.

        Returns:
            float: Current gain value
        """
        with self._settings_lock:
            return self._get_gain()

    @property
    def gain_set(self) -> float | str:
        """
        Thread-safe getter for current gain setting.

        Returns:
            float | str: Current gain setting
        """
        with self._settings_lock:
            return self._gain_set

    @property
    def frame_rate(self) -> float:
        """
        Get the current actual frame rate based on frame timing.

        Returns:
            float: Current frame rate in frames per second
        """
        return self._actual_frame_rate

    @property
    def bandwidth(self) -> float | int | str:
        """
        Thread-safe getter for current bandwidth value.

        Returns:
            Union[float, int, str]: Current bandwidth value
        """
        with self._settings_lock:
            return self._bandwidth_val

    @property
    def frame_res(self) -> Tuple[int, int]:
        """
        Get the camera resolution.

        Returns:
            tuple: Camera resolution as (width, height)
        """
        return self.FRAME_RES

    @property
    def sensor_size(self) -> Tuple[float, float]:
        """
        Get the camera sensor size.

        Returns:
            tuple: Camera sensor size as (width_mm, height_mm)
        """
        return self.SENSOR_SIZE

    @property
    def dtype(self) -> str:
        """
        Get the data type of the image data.

        Returns:
            str: Data type of the image data
        """
        return self.DTYPE


class CameraStream:
    name: str
    _cam: CameraInterface
    _cam_mdl: FixedZoomCamera | None
    _save_root_dir: pathlib.Path
    _save_dir: pathlib.Path
    _capture_killswitch: threading.Event = threading.Event()
    _save_killswitch: threading.Event = threading.Event()
    _save_queue: queue.Queue = queue.Queue()
    _capture_thread: threading.Thread | None = None
    _save_thread: threading.Thread | None = None
    _frame_count: int = 0
    _latest_frame: Frame | None = None
    _frame_lock: threading.RLock = threading.RLock()
    frame_available_event: threading.Event = threading.Event()
    _save_enabled: bool = False
    _mdata_fnames = [
        "timestamp",
        "time_string",
        "camera_name",
        "camera_model",
        "frame_rate",
        "save_time",
        "gain",
        "exposure",
        "frame_num",
    ]

    def __init__(
        self,
        name: str,
        cam_ifc: CameraInterface,
        save_root_dir: str,
        foc_len_mm: float | None = None,
    ):
        """
        Initialise a threaded camera stream manager.

        Args:
            camera: An instance of a Camera class that implements the Camera interface
            save_dir: Optional directory to save images
        """

        self.name = name
        self._cam = cam_ifc
        if foc_len_mm is not None:
            self.cam_mdl = FixedZoomCamera(
                res=self._cam.frame_res,
                sensor_size=self._cam.sensor_size,
                focal_length=foc_len_mm,
            )
        else:
            self.cam_mdl = None

        self.save_root_dir = pathlib.Path(save_root_dir).expanduser()
        self.save_dir = os.path.join(
            self.save_root_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        # Flags
        self._save_enabled = False

        # Threads
        # TODO: Add limit to queue
        self._save_queue = queue.Queue(maxsize=500)

        # Frame counter for saving
        self.frame_count = 0

        # Thread-safe latest frame
        self._frame_lock = threading.RLock()

        # Event to signal new frame availability
        self.frame_available_event = threading.Event()

    @property
    def cam(self) -> CameraInterface:
        """Get the camera interface."""
        return self._cam

    @property
    def save_queue_length(self) -> int:
        """Get the current length of the save queue."""
        return self._save_queue.qsize()

    def latest_frame(self) -> Frame | None:
        """
        Thread-safe access to the most recent frame.

        Returns:
            The most recent frame as a Frame object, or None if no frame is available
        """

        if self.frame_available_event.wait(timeout=1):
            with self._frame_lock:
                return self._latest_frame
                self.frame_available_event.clear()
        else:
            return None

    def __enter__(self):
        """Start the camera capture and save threads."""

        self._capture_killswitch.clear()
        try:
            self._cam = self.cam.__enter__()
            # Start capture thread
            self._capture_thread = threading.Thread(target=self._capture_loop)
            # self._capture_thread.daemon = True
            self._capture_thread.start()
        except:
            try:
                self._cam.__exit__()
                self._capture_killswitch.set()
                self._capture_thread.join()
            except:
                pass
            raise RuntimeError(f"Could not find {self.name}")
        return self

    # def stop(self):
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop the camera capture and save threads."""
        try:
            self._capture_killswitch.set()
            if self._capture_thread:
                self._capture_thread.join()
            if self._save_thread:
                self._save_thread.join()
            print(self.name + " stream gracefully closed")
        except Exception as e:
            print(f"Error gracefully closing camera stream: {e}")
        self.cam.__exit__(exc_type, exc_value, traceback)

    def _capture_loop(self):
        """Main loop for capturing frames from the camera."""
        while not self._capture_killswitch.is_set():
            try:
                # Capture frame - this will automatically update frame rate statistics
                frame = self.cam.get_frame()

                # Update latest frame with thread safety
                with self._frame_lock:
                    self._latest_frame = frame
                    self.frame_available_event.set()  # Signal that a new frame is available

                # Queue frame for saving if enabled
                if self._save_enabled:
                    try:
                        self._save_queue.put((frame, self._frame_count), block=False)
                    except queue.Full:
                        try:
                            old_frame, _ = self._save_queue.get_nowait()
                            delta_t = frame.timestamp - old_frame.timestamp
                            print(
                                f"Warning: Save queue full, dropping frame from {delta_t:.2f}s ago"
                            )
                        except queue.Empty:
                            continue  # Should not happen, but just in case
                        try:
                            self._save_queue.put((frame, self._frame_count))
                        except queue.Full:
                            pass  # Give up on this frame
                    self._frame_count += 1

            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Brief pause on error

    def _save_loop(self):
        """Main loop for saving frames to disk."""

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        mdata_save_path = os.path.join(self.save_dir, f"metadata_log.csv")
        with open(mdata_save_path, "a", newline="") as meta_log_file:
            writer = csv.DictWriter(meta_log_file, fieldnames=self._mdata_fnames)

            while not self._save_killswitch.is_set():
                try:
                    # Get frame from save queue
                    frame, frame_num = self._save_queue.get(timeout=1.0)
                    pix = frame.pixels.view(
                        frame.pixels.dtype.type
                    )  # Remove metadata view
                    save_metadata = {
                        "timestamp": frame.timestamp,
                        "time_string": datetime.datetime.fromtimestamp(
                            frame.timestamp
                        ).isoformat(),
                        "camera_name": self.name,
                        "camera_model": self.cam.name,
                        "frame_rate": self.cam.frame_rate,
                        "save_time": str(time.time()),
                        "gain": frame.gain,
                        "exposure": frame.exposure,
                        "frame_num": frame_num,
                    }

                    # Save frame with metadata
                    img_save_path = os.path.join(
                        self.save_dir,
                        f"f_{frame_num:06d}_{self.name}_{frame.timestamp:.2f}.npy",
                    )
                    np.save(
                        img_save_path,
                        pix,
                    )
                    writer.writerow(save_metadata)
                    meta_log_file.flush()

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in save loop: {e}")
                    time.sleep(0.1)  # Brief pause on error

    @property
    def save_enabled(self) -> bool:
        """
        Thread-safe getter for the save_enabled property.

        Returns:
            bool: True if saving is enabled, False otherwise.
        """
        with self._frame_lock:
            return self._save_enabled

    @save_enabled.setter
    def save_enabled(self, value: bool) -> None:
        """
        Thread-safe setter for the save_enabled property.

        Args:
            value (bool): True to enable saving, False to disable it.
        """
        with self._frame_lock:
            self._save_enabled = value

            t = self._save_thread
            if value and not (isinstance(t, threading.Thread) and t.is_alive()):
                # Create the save directory if it doesn't exist

                self._save_killswitch.clear()
                self._save_thread = threading.Thread(target=self._save_loop)
                self._save_thread.daemon = True
                self._save_thread.start()

            elif not value and (isinstance(t, threading.Thread) and t.is_alive()):
                self._save_killswitch.set()
                t.join()
                self._save_thread = None
