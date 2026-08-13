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
        self.monitor_rotation_deg = 90
        self.monitor_flip_x = False
        self.monitor_flip_y = False
        self.binning_horizontal = None
        self.binning_vertical = None
        self.binning_mode = None
        self._software_trigger_enabled = True
        self._frame_delivered = threading.Event()

    def _normalised_monitor_rotation(self) -> int:
        rot = int(getattr(self, "monitor_rotation_deg", 0) or 0) % 360
        if rot not in (0, 90, 180, 270):
            raise ValueError(
                f"Unsupported monitor rotation {rot}°. Use one of: 0, 90, 180, 270"
            )
        return rot

    def _rotate_for_monitoring(self, img):
        rot = self._normalised_monitor_rotation()
        if rot == 0:
            rotated = img
        elif rot == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if bool(getattr(self, "monitor_flip_x", False)):
            rotated = cv2.flip(rotated, 1)
        if bool(getattr(self, "monitor_flip_y", False)):
            rotated = cv2.flip(rotated, 0)
        return rotated


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

    def _apply_pixel_format(self) -> None:
        """Apply requested pixel format with robust fallbacks.

        Some camera/firmware combinations raise InternalFault when setting a
        valid-looking pixel format enum. In that case we fall back to the
        camera's current format (or first supported format) instead of failing
        camera connection.
        """
        requested_name = str(self.pixel_format)
        requested = self._resolve_pixel_format(requested_name)

        supported_formats = []
        if hasattr(self._vmbcam, "get_pixel_formats"):
            try:
                supported_formats = list(self._vmbcam.get_pixel_formats())
            except Exception:
                supported_formats = []

        candidate_formats: list[object] = [requested]

        current_format = None
        if hasattr(self._vmbcam, "get_pixel_format"):
            try:
                current_format = self._vmbcam.get_pixel_format()
            except Exception:
                current_format = None
        if current_format is not None:
            candidate_formats.append(current_format)

        for fmt in supported_formats:
            if fmt not in candidate_formats:
                candidate_formats.append(fmt)

        last_exc: Exception | None = None
        for fmt in candidate_formats:
            try:
                self._vmbcam.set_pixel_format(fmt)
                chosen_name = self._pixel_format_name(fmt)
                if chosen_name != requested_name:
                    warnings.warn(
                        f"Could not set requested pixel format '{requested_name}' on camera {self.cam_id}; "
                        f"using '{chosen_name}' instead."
                    )
                self.pixel_format = chosen_name
                self.DTYPE = self._pixel_dtype_for_format(chosen_name)
                return
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(
            f"Failed to set pixel format '{requested_name}' for camera {self.cam_id}."
        ) from last_exc

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
            warnings.warn(
                f"Camera {self.cam_id} does not expose SensorBitDepth; continuing without setting it."
            )
            return

        try:
            self._vmbcam.SensorBitDepth.set(self.sensor_bit_depth)
        except Exception as exc:
            current_value = None
            try:
                current_value = self._vmbcam.SensorBitDepth.get()
            except Exception:
                pass

            if current_value is not None:
                warnings.warn(
                    f"Could not set sensor bit depth '{self.sensor_bit_depth}' for camera {self.cam_id}. "
                    f"Current sensor bit depth is '{current_value}'. Continuing."
                )
                return

            warnings.warn(
                f"Failed to set sensor bit depth '{self.sensor_bit_depth}' for camera {self.cam_id} "
                f"({type(exc).__name__}: {exc}). Continuing."
            )
            return

    @staticmethod
    def _normalise_binning_mode(mode: object) -> str | None:
        if mode in [None, "", "default"]:
            return None

        mode_str = str(mode).strip().lower()
        if mode_str == "sum":
            return "Sum"
        if mode_str == "average":
            return "Average"
        raise ValueError(
            f"Unsupported binning mode '{mode}'. Use 'sum' or 'average'."
        )

    def _set_first_available_feature(self, feature_names: list[str], value: object) -> bool:
        for feature_name in feature_names:
            feature = getattr(self._vmbcam, feature_name, None)
            if feature is None:
                continue
            try:
                feature.set(value)
                return True
            except Exception:
                continue
        return False

    def _apply_binning(self) -> None:
        if self.binning_horizontal in [None, "", 1] and self.binning_vertical in [None, "", 1]:
            return

        if self.binning_horizontal not in [None, "", 1]:
            if not self._set_first_available_feature(["BinningHorizontal"], int(self.binning_horizontal)):
                warnings.warn(f"Camera {self.cam_id} does not expose BinningHorizontal")

        if self.binning_vertical not in [None, "", 1]:
            if not self._set_first_available_feature(["BinningVertical"], int(self.binning_vertical)):
                warnings.warn(f"Camera {self.cam_id} does not expose BinningVertical")

        mode = self._normalise_binning_mode(self.binning_mode)
        if mode is None:
            return

        if self._set_first_available_feature(
            ["BinningMode", "BinningHorizontalMode", "BinningVerticalMode"],
            mode,
        ):
            return

        selector_feature = getattr(self._vmbcam, "BinningSelector", None)
        mode_feature = getattr(self._vmbcam, "BinningMode", None)
        if selector_feature is not None and mode_feature is not None:
            for selector in ("Horizontal", "Vertical", "All"):
                try:
                    selector_feature.set(selector)
                    mode_feature.set(mode)
                except Exception:
                    continue
            return

        warnings.warn(
            f"Camera {self.cam_id} does not expose a supported binning mode feature"
        )

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

        try:
            self._vmbcam.stop_streaming()
        except Exception:
            pass
        self._apply_sensor_bit_depth()
        try:
            self._apply_pixel_format()
        except Exception as exc:
            warnings.warn(
                f"Could not apply requested pixel format '{self.pixel_format}' on camera {self.cam_id} "
                f"({type(exc).__name__}: {exc}). Continuing with camera default/current format."
            )
            try:
                current_format = self._vmbcam.get_pixel_format()
                current_name = self._pixel_format_name(current_format)
                self.pixel_format = current_name
                self.DTYPE = self._pixel_dtype_for_format(current_name)
            except Exception:
                pass
        self._apply_binning()

        # Read actual resolution from hardware — overrides the class-level constant
        # so that monitoring geometry (crosshairs, scale factor) is always correct.
        try:
            w = int(self._vmbcam.Width.get())
            h = int(self._vmbcam.Height.get())
        except Exception as exc:
            w, h = self.FRAME_RES
            warnings.warn(
                f"Could not read Width/Height on camera {self.cam_id} ({type(exc).__name__}: {exc}); "
                f"using fallback FRAME_RES={self.FRAME_RES}."
            )
        self.FRAME_RES = (w, h)
        try:
            self._vmbcam.DeviceLinkThroughputLimit.set(400e6)
        except Exception:
            pass

        self._limits = {
            "exposure": (1.0, 1e7),
            "exposure_incr": 1.0,
            "gain": (0.0, 100.0),
            "gain_incr": 0.1,
        }
        try:
            self._limits["exposure"] = self._vmbcam.ExposureTime.get_range()
            self._limits["exposure_incr"] = self._vmbcam.ExposureTime.get_increment()
        except Exception:
            pass
        try:
            self._limits["gain"] = self._vmbcam.Gain.get_range()
            self._limits["gain_incr"] = self._vmbcam.Gain.get_increment()
        except Exception:
            pass

        try:
            self.set_exposure(self._exposure_set if self._exposure_set else self.EXP_DEFAULT)
        except Exception as exc:
            warnings.warn(
                f"Could not set startup exposure on camera {self.cam_id} "
                f"({type(exc).__name__}: {exc}). Continuing."
            )
        try:
            self.set_gain(self._gain_set if self._gain_set else self.GAIN_DEFAULT)
        except Exception as exc:
            warnings.warn(
                f"Could not set startup gain on camera {self.cam_id} "
                f"({type(exc).__name__}: {exc}). Continuing."
            )
        self.start_video()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context related to this object."""
        if hasattr(self, "_vmbcam"):
            try:
                self._vmbcam.stop_streaming()
            except Exception:
                pass
            try:
                self._vmbcam.TriggerMode.set("Off")
            except Exception:
                pass
            try:
                self._vmbcam.AcquisitionMode.set("SingleFrame")
            except Exception:
                pass
            try:
                self._vmbcam.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
            print(f"Alvium {self.MODEL_NO} camera gracefully disconnected.")
        if hasattr(self, "_vmb"):
            try:
                self._vmb.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass

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

        try:
            self._vmbcam.stop_streaming()
        except Exception:
            pass

        self._software_trigger_enabled = True
        try:
            self._vmbcam.TriggerSource.set("Software")
            self._vmbcam.TriggerSelector.set("FrameStart")
            self._vmbcam.TriggerMode.set("On")
        except Exception as exc:
            self._software_trigger_enabled = False
            warnings.warn(
                f"Could not enable software trigger on camera {self.cam_id} "
                f"({type(exc).__name__}: {exc}); falling back to free-run mode."
            )
            try:
                self._vmbcam.TriggerMode.set("Off")
            except Exception:
                pass

        try:
            self._vmbcam.AcquisitionMode.set("Continuous")
        except Exception:
            pass
        self._vmbcam.start_streaming(frame_handler)

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
        except Exception:
            if isinstance(self._gain_set, (int, float)):
                gain = float(self._gain_set)
            else:
                gain = 0.0
        try:
            exposure = self.exposure
        except Exception:
            exposure = float(self._exposure_set) if self._exposure_set else 0.0

        self._frame_delivered.clear()
        timestamp = time.time()
        if self._software_trigger_enabled:
            try:
                self._vmbcam.TriggerSoftware.run()
            except Exception as exc:
                self._software_trigger_enabled = False
                warnings.warn(
                    f"TriggerSoftware failed on camera {self.cam_id} "
                    f"({type(exc).__name__}: {exc}); switching to free-run capture."
                )
        if self._frame_delivered.wait(timeout=2.0):
            actual_time = timestamp + (exposure / 2) / 1e6
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
        pix = self._rotate_for_monitoring(frame.pixels)
        return Frame(pix, frame.gain, frame.exposure, frame.timestamp, frame.cam_name)

    @property
    def monitoring_frame_res(self) -> tuple[int, int]:
        """Resolution after convert_for_monitoring (width, height)."""
        w, h = self.FRAME_RES
        rot = self._normalised_monitor_rotation()
        if rot in (90, 270):
            return (h, w)
        return (w, h)

    def convert_mask_for_monitoring(self, mask):
        if mask.dtype == bool:
            rotated = self._rotate_for_monitoring(mask.astype("uint8"))
            return rotated.astype(bool)
        return self._rotate_for_monitoring(mask)


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


class Alvium812UV(Alvium811):
    NAME = "Alvium_812_UV"
    MODEL_NO = "812"
    FRAME_RES = (2848, 2848)
    SENSOR_SIZE = (2848*2.74*1e-3, 2848*2.74*1e-3)


class AlviumAny(Alvium811):
    """Connect to the first available Alvium camera, whatever model it is.

    Resolution and sensor size are read from the camera hardware after
    connecting, so no model-specific constants are needed.
    """
    NAME = "Alvium"
    MODEL_NO = ""          # empty string is always `in` any model string
    FRAME_RES = (1, 1)     # placeholder — overwritten in __enter__
    SENSOR_SIZE = (1.0, 1.0)

    def __enter__(self) -> "AlviumAny":
        super().__enter__()
        # Read actual resolution and pixel size from the connected camera
        w = int(self._vmbcam.Width.get())
        h = int(self._vmbcam.Height.get())
        self.FRAME_RES = (w, h)
        try:
            # SensorPixelSize is in µm on most Alvium models
            px_um = float(self._vmbcam.SensorPixelSize.get())
        except Exception:
            px_um = 3.45  # sensible fallback
        self.SENSOR_SIZE = (w * px_um * 1e-3, h * px_um * 1e-3)
        self.NAME = f"Alvium_{self._vmbcam.get_model()}"
        return self


