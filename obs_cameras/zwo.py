import threading
import zwoasi as asi
from typing import Any
import time
import warnings
from obs_cameras.base import CameraInterface, Frame
from importlib.resources import files
import numpy as np
import cv2

ASI_LIB = str(files("obs_cameras").joinpath("assets", "libASICamera2.so"))
asi.init(ASI_LIB)


class ASI585(CameraInterface):
    NAME = "ASI585"
    MODEL_NO = "585"
    FRAME_RES = (3840, 2160)
    SENSOR_SIZE = (11.136, 6.264)
    USB_TRANSFER_TIME: float = 0.001  # Estimated USB transfer time in seconds (1ms)
    DTYPE = "uint8"
    GAIN_DEFAULT = 5
    EXP_DEFAULT = 20e3

    _frame_delivered = threading.Event
    _asicam: asi.Camera
    _controls: dict[str, dict[str, Any]]

    _limits = {}

    def __init__(self):
        super().__init__()
        self.cam_id = None
        self._frame_delivered = threading.Event()

    @property
    def name(self) -> str:
        return self.NAME

    def reconnect(self, idx=0) -> bool:
        cam_dict, num_cameras = self.list_devices()
        if num_cameras == 0:
            print("No cameras found")
            return False
        elif num_cameras > 1:
            print("Setting 'ASI' to num " + str(idx))
        self._asicam = asi.Camera(idx)
        return True

    def list_devices(self) -> tuple[dict, int]:
        num_cameras = asi.get_num_cameras()
        cameras_found = asi.list_cameras()
        return cameras_found, num_cameras

    def start_video(self) -> None:
        self._asicam.stop_exposure()
        self._asicam.set_control_value(14, 1)  # Set high speed mode
        self._asicam.start_video_capture()

    def __enter__(self) -> CameraInterface:
        if not self.reconnect():
            raise RuntimeError("No ASI cameras found")
        controls = self._asicam.get_controls()
        self._limits = {
            "exposure": (
                controls["Exposure"]["MinValue"],
                controls["Exposure"]["MaxValue"],
            ),
            "gain": (controls["Gain"]["MinValue"], controls["Gain"]["MaxValue"]),
            "gain_default": controls["Gain"]["DefaultValue"],
            "bandwidth": (controls["BandWidth"]["MinValue"], controls["BandWidth"]["MaxValue"]),
        }

        self.start_video()
        self._asicam.set_image_type(asi.ASI_IMG_RAW8)
        self.set_exposure(self.EXP_DEFAULT)
        self.set_gain(self.GAIN_DEFAULT)
        self.set_bandwidth("auto")
        self.start_video()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._asicam.stop_video_capture()
        self._asicam.close()

    def _get_frame(self) -> Frame:
        """
        Hardware-specific implementation of frame capture.
        Must be implemented by derived classes.

        Returns:
            Frame: A Frame object containing the image data and metadata
            timestamp: Timestamp when the frame was captured
        """
        try:
            gain = self.gain
            exposure = self.exposure
            frame = self._asicam.capture_video_frame(timeout=1000)
            capture_time = time.time()
            adjusted_time = (
                capture_time - (self.exposure / 2) / 1e6 - self.USB_TRANSFER_TIME
            )
            return Frame(frame, gain, exposure, adjusted_time, self.name)
        except asi.ZWO_IOError:
            raise RuntimeError(f"Cam: {self.name} Frame capture timed out.")

    def _get_gain(self) -> float:
        """
        Hardware-specific implementation of gain retrieval.
        Must be implemented by derived classes.

        Returns:
            float: Current gain value
        """
        return self._asicam.get_control_value(asi.ASI_GAIN)[0]

    def _get_exposure(self) -> float:
        """
        Hardware-specific implementation of exposure retrieval.
        Must be implemented by derived classes.

        Returns:
            float: Current exposure time in microseconds
        """
        return self._asicam.get_control_value(asi.ASI_EXPOSURE)[0]

    def _set_exposure(self, exp: float) -> None:
        """
        Args:
            exp: Exposure time in microseconds
        """
        # if 0 < exp - self.exposure < self._limits["exposure_incr"]:
        #     exp = self.exposure + self._limits["exposure_incr"]
        # elif 0 > exp - self.exposure > -self._limits["exposure_incr"]:
        #     exp = self.exposure - self._limits["exposure_incr"]

        if not self._limits["exposure"][0] <= exp <= self._limits["exposure"][1]:
            print("Clipping exposure to valid range.")
        exp = max(self._limits["exposure"][0], min(self._limits["exposure"][1], exp))
        self._asicam.set_control_value(asi.ASI_EXPOSURE, int(exp))

    def _set_gain(self, gain: float | int | str) -> None:
        """
        Args:
            gain: Gain value
        """
        if isinstance(gain, float | int):
            gain = int(gain)
            if not self._limits["gain"][0] <= gain <= self._limits["gain"][1]:
                print("Clipping gain to valid range.")
            gain = max(self._limits["gain"][0], min(self._limits["gain"][1], gain))
            self._asicam.set_control_value(asi.ASI_GAIN, int(gain))

        elif gain == "auto":
            self._asicam.set_control_value(
                asi.ASI_GAIN,
                self._limits["gain_default"],
                auto=True,
            )
        else:
            warnings.warn("Gain value not recognised; no changes made.")

    def _set_bandwidth(self, bw: float | str) -> None:
        """
        Args:
            bw: Bandwidth limit in Mbps
        """
        if isinstance(bw, float):
            if not self._limits["bandwidth"][0] <= bw <= self._limits["bandwidth"][1]:
                print("Clipping bandwidth to valid range.")
            bw = max(self._limits["bandwidth"][0], min(
                self._limits["bandwidth"][1], bw))
            self._asicam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, int(bw))

        if bw == "auto":
            self._asicam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 80, auto=True)

        if bw == "min":
            self._asicam.set_control_value(
                asi.ASI_BANDWIDTHOVERLOAD,
                int(self._limits["bandwidth"][0])
            )
        elif bw == "min":
            self._asicam.set_control_value(
                asi.ASI_BANDWIDTHOVERLOAD,
                int(self._limits["bandwidth"][1]),
            )
        else:
            warnings.warn("Bandwidth value not recognised; no changes made.")


    def convert_for_monitoring(self, frame: Frame) -> Frame:
        """
        Convert the frame for monitoring purposes.
        This method can be overridden by derived classes if needed.

        Args:
            frame (Frame): The frame to convert.

        Returns:
            Frame: The converted frame.
        """
        pix = frame.pixels
        pix = cv2.cvtColor(pix, cv2.COLOR_BayerRGGB2BGR)
        pix = cv2.rotate(pix, cv2.ROTATE_90_CLOCKWISE)
        # pix = cv2.flip(pix, 0)
        # pix = cv2.flip(pix, 1)
        # pix = np.flip(np.transpose(pix, (1, 0, 2)), 1)
        return Frame(pix, frame.gain, frame.exposure, frame.timestamp, frame.cam_name)
