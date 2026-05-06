import time
import vmbpy
import threading
import warnings
import cv2
from .base import CameraInterface, Frame


class Alvium811(CameraInterface):
    NAME = "Alvium_811"
    MODEL_NO = "811"
    FRAME_RES = (2848, 2848)
    SENSOR_SIZE = (2848*2.74*1e-3, 2848*2.74*1e-3)
    DTYPE = "uint8"
    PIXEL_FORMAT = "Mono8"
    SENSOR_BIT_DEPTH = None
    GAIN_DEFAULT = 1
    EXP_DEFAULT = 20

    _vmb: vmbpy.VmbSystem
    _vmbcam: vmbpy.Camera
    cam_id: str | None
    _frame: vmbpy.Frame
    _frame_delivered: threading.Event

    _limits = {}

    def __init__(self):
        super().__init__()
        self.cam_id = None
        self.pixel_format = self.PIXEL_FORMAT
        self.sensor_bit_depth = self.SENSOR_BIT_DEPTH
        self._frame_delivered = threading.Event()


    @property
    def name(self) -> str:
        return self.NAME

    def reconnect(self):
        cam_dict = self.list_devices()
        vmbcam = None

        if self.cam_id is None:
            try:
                vmbcam = [
                    cam
                    for cam in cam_dict.keys()
                    if self.MODEL_NO in cam_dict[cam]["Model"]
                ][0]
            except IndexError:
                warnings.warn(f"No Alvium {self.MODEL_NO} camera found.")
        else:
            try:
                vmbcam = [
                    cam for cam in cam_dict.keys() if self.cam_id in cam_dict[cam]["ID"]
                ][0]
            except IndexError:
                warnings.warn(f"Camera with ID {self.cam_id} not found.")
        if vmbcam is not None:
            self._vmbcam = vmbcam
            self.cam_id = cam_dict[vmbcam]["ID"]

    def list_devices(self) -> dict:
        cam_dict = {}
        cams = self._vmb.get_all_cameras()
        for cam in cams:
            try:
                with cam:
                    cam_dict[cam] = {"Model": cam.get_model(), "ID": cam.get_id()}
            except vmbpy.error.VmbCameraError:
                pass
        return cam_dict

    def _resolve_pixel_format(self, pixel_format_name: str) -> vmbpy.PixelFormat:
        try:
            return getattr(vmbpy.PixelFormat, pixel_format_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported Alvium pixel format: {pixel_format_name}") from exc

    def _pixel_format_name(self, pixel_format: object) -> str:
        name = getattr(pixel_format, "name", None)
        if isinstance(name, str):
            return name

        pixel_format_str = str(pixel_format)
        if "." in pixel_format_str:
            return pixel_format_str.rsplit(".", 1)[-1]
        return pixel_format_str

    def _validate_supported_pixel_format(
        self, requested_pixel_format: vmbpy.PixelFormat
    ) -> None:
        supported_formats = []
        if hasattr(self._vmbcam, "get_pixel_formats"):
            supported_formats = list(self._vmbcam.get_pixel_formats())

        if supported_formats and requested_pixel_format not in supported_formats:
            supported_names = ", ".join(
                self._pixel_format_name(pixel_format)
                for pixel_format in supported_formats
            )
            raise ValueError(
                f"Pixel format '{self.pixel_format}' is not supported by camera {self.cam_id}. "
                f"Supported formats: {supported_names}"
            )

    def _pixel_dtype_for_format(self, pixel_format_name: str) -> str:
        if pixel_format_name == "Mono8":
            return "uint8"
        if pixel_format_name.startswith("Mono"):
            return "uint16"
        return self.DTYPE

    def _apply_sensor_bit_depth(self) -> None:
        if self.sensor_bit_depth in [None, "", "default"]:
            return

        if not hasattr(self._vmbcam, "SensorBitDepth"):
            raise ValueError(
                f"Camera {self.cam_id} does not expose a SensorBitDepth feature"
            )

        try:
            self._vmbcam.SensorBitDepth.set(self.sensor_bit_depth)
        except Exception as exc:
            current_value = None
            try:
                current_value = self._vmbcam.SensorBitDepth.get()
            except Exception:
                pass

            if current_value is not None:
                raise ValueError(
                    f"Unsupported sensor bit depth '{self.sensor_bit_depth}' for camera {self.cam_id}. "
                    f"Current sensor bit depth is '{current_value}'."
                ) from exc
            raise ValueError(
                f"Failed to set sensor bit depth '{self.sensor_bit_depth}' for camera {self.cam_id}."
            ) from exc

    def __enter__(self) -> CameraInterface:
        """Enter the runtime context related to this object."""
        _vmb = vmbpy.VmbSystem.get_instance()
        self._vmb = _vmb.__enter__()
        time.sleep(0.1)

        self.reconnect()
        if hasattr(self, "_vmbcam"):
            self._vmbcam.__enter__()
        else:
            self.__exit__(None, None, None)
            raise RuntimeError("Failed to connect to camera.")

        self._vmbcam.stop_streaming()
        self._apply_sensor_bit_depth()
        requested_pixel_format = self._resolve_pixel_format(self.pixel_format)
        self._validate_supported_pixel_format(requested_pixel_format)
        self._vmbcam.set_pixel_format(requested_pixel_format)
        self.DTYPE = self._pixel_dtype_for_format(self.pixel_format)
        self._vmbcam.DeviceLinkThroughputLimit.set(400e6)
        self._limits = {
            "exposure": self._vmbcam.ExposureTime.get_range(),
            "exposure_incr": self._vmbcam.ExposureTime.get_increment(),
            "gain": self._vmbcam.Gain.get_range(),
            "gain_incr": self._vmbcam.Gain.get_increment(),
        }
        self.set_exposure(self.EXP_DEFAULT)
        self.set_gain(self.GAIN_DEFAULT)
        self.start_video()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context related to this object."""
        if hasattr(self, "_vmbcam"):
            self._vmbcam.stop_streaming()
            self._vmbcam.TriggerMode.set("Off")
            self._vmbcam.AcquisitionMode.set("SingleFrame")
            self._vmbcam.__exit__(exc_type, exc_val, exc_tb)
            print(f"Alvium {self.MODEL_NO} camera gracefully disconnected.")
        self._vmb.__exit__(exc_type, exc_val, exc_tb)

    def start_video(self) -> None:
        """
        Start video capture mode.
        Must be implemented by derived classes.
        """

        # Think this is dodgy? Assigning to self when that isn't a function argument?
        def frame_handler(cam: vmbpy.Camera, stream: vmbpy.Stream, frame: vmbpy.Frame):
            self._frame = frame.as_numpy_ndarray()
            self._frame_delivered.set()
            cam.queue_frame(frame)

        self._vmbcam.stop_streaming()
        self._vmbcam.TriggerSource.set("Software")
        self._vmbcam.TriggerSelector.set("FrameStart")
        self._vmbcam.TriggerMode.set("On")
        self._vmbcam.AcquisitionMode.set("Continuous")
        self._vmbcam.start_streaming(frame_handler)

    def _get_frame(self) -> Frame:
        """
        Hardware-specific implementation of frame capture.
        Must be implemented by derived classes.

        Returns:
            Frame: A Frame object containing the image data and metadata
            timestamp: Timestamp when the frame was captured
        """
        gain = self.gain
        exposure = self.exposure
        self._frame_delivered.clear()
        timestamp = time.time()
        self._vmbcam.TriggerSoftware.run()
        if self._frame_delivered.wait(timeout=2.0):
            actual_time = timestamp + (self.exposure / 2) / 1e6
            frame = Frame(
                pixels=self._frame,
                gain=gain,
                exposure=exposure,
                timestamp=actual_time,
                cam_name=self.name,
            )
            return frame
        else:
            raise RuntimeError(f"Cam {self.name} frame capture timed out.")

    def _get_gain(self) -> float:
        """
        Hardware-specific implementation of gain retrieval.
        Must be implemented by derived classes.

        Returns:
            float: Current gain value
        """
        return self._vmbcam.Gain.get()

    def _get_exposure(self) -> float:
        """
        Hardware-specific implementation of exposure retrieval.
        Must be implemented by derived classes.

        Returns:
            float: Current exposure time in microseconds
        """
        return self._vmbcam.ExposureTime.get() / 1e3

    def _set_exposure(self, exp: float) -> None:
        """
        Args:
            exp: Exposure time in milliseconds
        """
        exp = int(exp*1e3)  # Convert to microseconds for camera API
        if 0 < exp - self.exposure < self._limits["exposure_incr"]:
            exp = self.exposure + self._limits["exposure_incr"]
        elif 0 > exp - self.exposure > -self._limits["exposure_incr"]:
            exp = self.exposure - self._limits["exposure_incr"]

        if not self._limits["exposure"][0] <= exp <= self._limits["exposure"][1]:
            print("Clipping exposure to valid range.")
        exp = max(self._limits["exposure"][0], min(self._limits["exposure"][1], exp))
        self._vmbcam.ExposureTime.set(exp)

    def _set_gain(self, gain: float | str) -> None:
        """
        Args:
            gain: Gain value
        """
        if isinstance(gain, float):
            if 0 < gain - self.gain < self._limits["gain_incr"]:
                gain = self.gain + self._limits["gain_incr"]
            elif 0 > gain - self.gain > -self._limits["gain_incr"]:
                gain = self.gain - self._limits["gain_incr"]

            if not self._limits["gain"][0] <= gain <= self._limits["gain"][1]:
                print("Clipping gain to valid range.")
            gain = max(self._limits["gain"][0], min(self._limits["gain"][1], gain))
            self._vmbcam.Gain.set(gain)

    def convert_for_monitoring(self, frame: Frame) -> Frame:
        # Convert to 8-bit grayscale for monitoring
        pix = frame.pixels
        # pix = (frame.pixels / 2**4).astype("uint8")
        # pix = cv2.resize(converted_frame, (self.DISPLAY_RES[1], self.DISPLAY_RES[0]))
        # pix = cv2.flip(pix, 1)
        pix = cv2.rotate(pix, cv2.ROTATE_90_CLOCKWISE)
        return Frame(pix, frame.gain, frame.exposure, frame.timestamp, frame.cam_name)
        self._vmbcam.Gain.set(gain)

    def convert_mask_for_monitoring(self, mask):
        if mask.dtype == bool:
            rotated = cv2.rotate(mask.astype("uint8"), cv2.ROTATE_90_CLOCKWISE)
            return rotated.astype(bool)
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)


class Alvium508(Alvium811):
    NAME = "Alvium_508"
    MODEL_NO = "508"
    FRAME_RES = (2464, 2056)
    SENSOR_SIZE = (2464*3.45*1e-3, 2056*3.45*1e-3)


class AlviumU130VSWIR(Alvium811):
    NAME = "Alvium_U130_VSWIR"
    MODEL_NO = "U-130"
    FRAME_RES = (1296, 1032)
    SENSOR_SIZE = (1296*5*1e-3, 1032*5*1e-3)


