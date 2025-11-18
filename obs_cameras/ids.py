from __future__ import annotations
import time
import threading
import warnings
import cv2
import ids_peak.ids_peak as ids
import ids_peak.ids_peak_ipl_extension as ids_ipl_extension

from obs_cameras.base import CameraInterface, Frame


class IDSU33080(CameraInterface):
    NAME = "IDS-U33080"
    MODEL_NO = "U33080"
    FRAME_RES = (2464, 2056)
    SENSOR_SIZE = (8.473, 7.086)  # in meters
    DTYPE = "uint12"
    GAIN_DEFAULT = 1
    EXP_DEFAULT = 20e3

    _idscam: ids.Camera
    _frame_delivered = threading.Event
    _limits = {}
    _rdn: ids.RemoteDeviceNodemap
    _datastream: ids.DataStream

    @property
    def name(self) -> str:
        return self.NAME

    def __init__(self):
        super().__init__()
        self._frame_delivered = threading.Event()
        ids.Library.Initialize()

    def reconnect(self, idx=0) -> bool:
        try:
            device_manager = ids.DeviceManager.Instance()
            device_manager.Update()
            device_descriptors = device_manager.Devices()
            for device_descriptor in device_descriptors:
                print(device_descriptor.DisplayName())
            self._idscam = device_manager.Devices()[idx].OpenDevice(
                ids.DeviceAccessType_Control
            )
            return True
        except Exception as e:
            warnings.warn(f"IDS camera connection failed: {e}")
            return False

    def __enter__(self) -> IDSU33080:
        if not self.reconnect():
            raise RuntimeError("No IDS cameras found")

        self._rdn = self._idscam.RemoteDevice().NodeMaps()[0]

        self._limits = {
            "exposure": (
                self._rdn.FindNode("ExposureTime").Minimum(),
                self._rdn.FindNode("ExposureTime").Maximum(),
            ),
            "gain": (
                self._rdn.FindNode("Gain").Minimum(),
                self._rdn.FindNode("Gain").Maximum(),
            ),
        }
        self.start_video()
        self.set_exposure(self.EXP_DEFAULT)
        self.set_gain(self.GAIN_DEFAULT)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._rdn.FindNode("AcquisitionStop").Execute()
        self._datastream.StopAcquisition()
        ids.Library.Close()

    def pause_video(self) -> None:
        self._rdn.FindNode("AcquisitionStop").Execute()
        self._rdn.FindNode("TriggerMode").SetCurrentEntry("Off")

    def resume_video(self) -> None:
        self._rdn.FindNode("TriggerSource").SetCurrentEntry("Software")
        self._rdn.FindNode("TriggerMode").SetCurrentEntry("On")
        self._rdn.FindNode("AcquisitionStart").Execute()
        self._rdn.FindNode("AcquisitionStart").WaitUntilDone()

    def _set_exposure(self, exp: float) -> None:
        exp = int(exp * 1e3)
        if not self._limits["exposure"][0] <= exp <= self._limits["exposure"][1]:
            print("Clipping exposure to valid range.")
        exp = max(self._limits["exposure"][0], min(self._limits["exposure"][1], exp))
        self.pause_video()
        self._rdn.FindNode("ExposureTime").SetValue(exp)
        self.resume_video()

    def _set_gain(self, gain: float | str) -> None:
        self._rdn.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
        if gain == "auto":
            self._rdn.FindNode("GainAuto").SetCurrentEntry("Continuous")
        elif isinstance(gain, float | int):
            if not self._limits["gain"][0] <= gain <= self._limits["gain"][1]:
                print("Clipping gain to valid range.")
            gain = max(self._limits["gain"][0], min(self._limits["gain"][1], gain))
            self._rdn.FindNode("GainAuto").SetCurrentEntry("Off")
            self._rdn.FindNode("Gain").SetValue(gain)
        else:
            warnings.warn("Gain value not recognised; no changes made.")

    def _get_exposure(self) -> float:
        return self._rdn.FindNode("ExposureTime").Value() / 1e3

    def _get_gain(self) -> float:
        self._rdn.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
        return self._rdn.FindNode("Gain").Value()

    def start_video(self) -> None:
        self._rdn.FindNode("PixelFormat").SetCurrentEntry("Mono12")
        self._rdn.FindNode("GainSelector").SetCurrentEntry("AnalogAll")

        self._datastream = self._idscam.DataStreams()[0].OpenDataStream()
        payload_size = self._rdn.FindNode("PayloadSize").Value()

        for _ in range(self._datastream.NumBuffersAnnouncedMinRequired()):
            buffer = self._datastream.AllocAndAnnounceBuffer(payload_size)
            self._datastream.QueueBuffer(buffer)

        # self._rdn.FindNode("AcquisitionFrameRate").SetValue(50)
        self._rdn.FindNode("TriggerSelector").SetCurrentEntry(
            "ExposureStart"
        )
        self._rdn.FindNode("TriggerSource").SetCurrentEntry("Software")
        self._rdn.FindNode("TriggerMode").SetCurrentEntry("On")

        self._datastream.StartAcquisition()
        self._rdn.FindNode("AcquisitionStart").Execute()
        self._rdn.FindNode("AcquisitionStart").WaitUntilDone()
        print("Video capture started")

    def _get_frame(self) -> Frame:

        self._rdn.FindNode("TriggerSoftware").Execute()
        trigger_time = time.time()

        buffer = self._datastream.WaitForFinishedBuffer(1000)
        raw_image = ids_ipl_extension.BufferToImage(buffer)
        self._datastream.QueueBuffer(buffer)
        picture = raw_image.get_numpy_3D_16()[:, :, 0].T

        capture_time = (
            trigger_time + (self.exposure / 2) / 1e6
        )
        return Frame(picture, self.gain, self.exposure, capture_time, self.name)

    def convert_for_monitoring(self, frame: Frame) -> Frame:
        # Convert to 8-bit grayscale for monitoring
        pix = (frame.pixels / 2**4).astype("uint8")
        # pix = cv2.resize(converted_frame, (self.DISPLAY_RES[1], self.DISPLAY_RES[0]))
        pix = cv2.flip(pix, 1)
        return Frame(pix, frame.gain, frame.exposure, frame.timestamp, frame.cam_name)
