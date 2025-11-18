import threading
import PyQt6
from PyQt6 import QtGui
import pyqtgraph as pg
import numpy as np
import cv2
import warnings
from dataclasses import dataclass
import datetime
from numpy.typing import NDArray
import time

import astrix as at

from obs_cameras.base import CameraStream
from obs_target.target import Target
from obs_utils.context import Context, State


@dataclass
class DisplaySettings:
    clahe_enabled: bool = False
    clahe_cliplimit: float = 2.0
    clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    norm_enabled: bool = False
    colourmap_enabled: bool = False
    hist_enabled: bool = False


class Display:
    _kill_event: threading.Event
    _new_stream_event: threading.Event
    _disp_lock: threading.RLock

    app: PyQt6.QtWidgets.QApplication
    win: pg.GraphicsLayoutWidget
    p1: pg.ViewBox
    img: pg.ImageItem
    labs: dict
    x_rules: pg.PlotDataItem
    y_rules: pg.PlotDataItem
    _scale_factor: float
    _display_res: tuple[int, int]
    _width: int

    _disp_set: DisplaySettings

    _ctx: Context
    _stream: CameraStream
    _cam_mdl: at.FixedZoomCamera
    _state: State
    _target: Target | None

    # _stream: CameraStream
    # _imu_monitor: CertusMonitor | None
    # _encoder_monitor: EncoderMonitor | None
    # _has_imu: bool
    # _has_encoder: bool
    # _overlay: Target | None
    # _has_overlay: bool

    def __init__(
        self,
        context: Context,
        state: State,
        target: Target | None,
    ):
        self._ctx = context
        self._stream = self._ctx.disp_stream
        self._state = state
        self._cam_mdl = self._stream.cam_mdl
        self._target = target

        self._width = 1920

        self._disp_lock = threading.RLock()
        self._kill_event = threading.Event()
        self._new_stream_event = threading.Event()
        self._disp_set = DisplaySettings()

        self.app = pg.Qt.mkQApp(name="Video-stream")
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle("Display " + str(self._stream.cam.name))
        self.win.keyPressEvent = self.on_key

        self.p1 = self.win.addViewBox(row=0, col=0)
        self.p1.setAspectLocked()

        img = pg.ImageItem()
        hist = pg.HistogramLUTItem(img, orientation="vertical")
        exp_lab = pg.TextItem()
        gain_lab = pg.TextItem()
        fps_lab = pg.TextItem()
        clahe_lab = pg.TextItem()
        saving_lab = pg.TextItem()
        save_queue = pg.TextItem()
        time_lab = pg.TextItem()
        gimb_angles_lab = pg.TextItem()
        target_angles_lab = pg.TextItem()
        imu_angles_lab = pg.TextItem()
        orient_cal_lab = pg.TextItem()
        track_alt_lab = pg.TextItem()

        self.win.addItem(hist, 0, 1)
        self.p1.addItem(img)
        self.p1.addItem(exp_lab)
        self.p1.addItem(gain_lab)
        self.p1.addItem(fps_lab)
        self.p1.addItem(clahe_lab)
        self.p1.addItem(saving_lab)
        self.p1.addItem(save_queue)
        self.p1.addItem(time_lab)
        self.p1.addItem(gimb_angles_lab)
        self.p1.addItem(target_angles_lab)
        self.p1.addItem(imu_angles_lab)
        self.p1.addItem(orient_cal_lab)
        self.p1.addItem(track_alt_lab)

        self.img = img
        self.labs = {
            "Exp": exp_lab,
            "Gain": gain_lab,
            "FPS": fps_lab,
            "CLAHE": clahe_lab,
            "Saving": saving_lab,
            "Save queue": save_queue,
            "Time": time_lab,
            "GIM": gimb_angles_lab,
            "IMU": imu_angles_lab,
            "TAR": target_angles_lab,
            "TrackAlt": track_alt_lab,
        }

        self.x_rules = pg.PlotDataItem()
        self.y_rules = pg.PlotDataItem()
        self.x_target = pg.PlotDataItem()
        self.y_target = pg.PlotDataItem()
        self.p1.addItem(self.x_rules)
        self.p1.addItem(self.y_rules)
        self.p1.addItem(self.x_target)
        self.p1.addItem(self.y_target)

        self.offset = 100
        self.traj_bounds_upper = pg.PlotDataItem()
        self.traj_bounds_lower = pg.PlotDataItem()
        self.traj_time_indicator = pg.PlotDataItem()
        self.p1.addItem(self.traj_bounds_upper)
        self.p1.addItem(self.traj_bounds_lower)
        self.p1.addItem(self.traj_time_indicator)

        self.set_bounds()
        self._init_stream()

    def on_key(self, ev: QtGui.QKeyEvent):
        txt = ev.text()
        if txt in ["s", "S"]:
            self._stream.save_enabled = not self._stream.save_enabled
        if txt == "G":
            old_gain = self._stream.cam.gain
            new_gain = max(old_gain * 1.1, old_gain + 1)
            self._stream.cam.set_gain(new_gain)
        elif txt == "g":
            old_gain = self._stream.cam.gain
            new_gain = min(old_gain * 0.9, old_gain - 1)
            self._stream.cam.set_gain(new_gain)
        elif txt == "E":
            old_exp = self._stream.cam.exposure
            new_exp = old_exp * 1.1
            self._stream.cam.set_exposure(new_exp)
        elif txt == "e":
            old_exp = self._stream.cam.exposure
            new_exp = old_exp * 0.9
            self._stream.cam.set_exposure(new_exp)
        elif txt in ["C", "c"]:
            self.set_clahe(not self._disp_set.clahe_enabled)
        elif txt in ["N", "n"]:
            self.set_norm(not self._disp_set.norm_enabled)
        elif txt in ["M", "m"]:
            self.set_colourmap(not self._disp_set.colourmap_enabled)
        elif txt in ["Q", "q"]:
            self.close()

    def set_bounds(self):
        # self.img_size = self._stream.cam.frame_res
        aspect = self._stream.cam.frame_res[0] / self._stream.cam.frame_res[1]
        height = self._width / aspect
        self._display_res = (int(self._width), int(height))
        self._scale_factor = self._display_res[0] / self._stream.cam.frame_res[0]

        # Set constant bounds for the view
        self.p1.setRange(
            xRange=(0, self._display_res[0]),
            yRange=(0, self._display_res[1]),
            padding=0,
        )

        # Disable auto-scaling
        self.p1.disableAutoRange()

        # Lock aspect ratio (optional)
        self.p1.setAspectLocked(True)

    def downscale_img(self, img):
        if self._scale_factor != 1.0:
            new_size = (
                int(img.shape[1] * self._scale_factor),
                int(img.shape[0] * self._scale_factor),
            )
            img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            return img_resized
        else:
            return img

    def signal_new_stream(self):
        self._new_stream_event.set()

    def change_stream(self):
        with self._disp_lock:
            self._stream = self._ctx.disp_stream
            self._cam_mdl = self._stream.cam_mdl
            self.set_bounds()
            self._init_stream()
        self._new_stream_event.clear()

    def _init_stream(self):
        img_ratio = self._display_res[0] / self._display_res[1]
        # win_dims = (
        #     int(self._width * 0.8),
        #     int(self._width * 0.8 * (1 / img_ratio) - 100),
        # )
        win_dims = (
            int(self._width * 1),
            int(self._width * 1 * (1 / img_ratio)),
        )
        self.win.resize(*win_dims)

        # Private re-scale function just for labels
        def rescale(x, y):
            if x > 0:
                x_ = x * self._display_res[0] / 1920
            else:
                x_ = self._display_res[0] + x * self._display_res[0] / 1920
            if y > 0:
                y_ = y * self._display_res[1] / 1080
            else:
                y_ = self._display_res[1] + y * self._display_res[1] / 1080
            return x_, y_

        self.labs["FPS"].setFont(QtGui.QFont("monospace", 12, 150))
        self.labs["FPS"].setColor("white")
        self.labs["FPS"].setPos(*rescale(20, 30))

        self.labs["CLAHE"].setFont(QtGui.QFont("monospace", 12, 150))
        self.labs["CLAHE"].setColor("white")
        self.labs["CLAHE"].setPos(*rescale(20, 105))

        self.labs["Gain"].setFont(QtGui.QFont("monospace", 12, 150))
        self.labs["Gain"].setColor("white")
        self.labs["Gain"].setPos(*rescale(20, 55))

        self.labs["Exp"].setFont(QtGui.QFont("monospace", 12, 150))
        self.labs["Exp"].setColor("white")
        self.labs["Exp"].setPos(*rescale(20, 80))

        self.labs["Saving"].setPos(*rescale(-250, -30))
        self.labs["Saving"].setFont(QtGui.QFont("monospace", 11, 700))
        self.labs["Saving"].setColor("r")

        self.labs["Save queue"].setPos(*rescale(-250, -55))
        self.labs["Save queue"].setFont(QtGui.QFont("monospace", 11, 700))
        self.labs["Save queue"].setColor("r")

        self.labs["Time"].setPos(*rescale(30, -30))
        self.labs["Time"].setFont(QtGui.QFont("monospace", 11, 100))
        self.labs["Time"].setColor("g")

        self.labs["GIM"].setPos(*rescale(-300, 30))
        self.labs["GIM"].setFont(QtGui.QFont("monospace", 11, 100))
        self.labs["GIM"].setColor("g")

        self.labs["IMU"].setPos(*rescale(-300, 55))
        self.labs["IMU"].setFont(QtGui.QFont("monospace", 11, 700))
        self.labs["IMU"].setColor("g")

        self.labs["TAR"].setPos(*rescale(-300, 80))
        self.labs["TAR"].setFont(QtGui.QFont("monospace", 11, 150))
        self.labs["TAR"].setColor("r")

        self.labs["TrackAlt"].setPos(*rescale(-300, 90))
        self.labs["TrackAlt"].setFont(QtGui.QFont("monospace", 11, 150))
        self.labs["TrackAlt"].setColor("g")

        self.crosshairs()

        self.app.processEvents()
        # self.win.showMaximized()
        self.win.setContentsMargins(0, 0, 0, 0)
        self.win.show()

    def set_clahe(self, enabled: bool, clip_limit: float = 2.0):
        with self._disp_lock:
            if enabled:
                self._disp_set.clahe_enabled = True
                self._disp_set.clahe = cv2.createCLAHE(
                    clipLimit=clip_limit, tileGridSize=(16, 16)
                )
            else:
                self._disp_set.clahe_enabled = False

    def set_norm(self, enabled: bool):
        with self._disp_lock:
            if enabled:
                self._disp_set.norm_enabled = True
            else:
                self._disp_set.norm_enabled = False

    def set_colourmap(self, enabled: bool):
        with self._disp_lock:
            if enabled:
                self._disp_set.colourmap_enabled = True
            else:
                self._disp_set.colourmap_enabled = False

    @staticmethod
    def soft_normalise(img, lower=0, upper=100):
        low_val = np.percentile(img, lower)
        high_val = np.percentile(img, upper)
        img_clipped = np.clip(img, low_val, high_val)
        norm_img = ((img_clipped - low_val) / (high_val - low_val) * 255).astype(
            np.uint8
        )
        return norm_img

    def update_img(self, img):
        """Update the image displayed in the window.
        Note: This img is assumed to be 8-bit grayscale.
        Args:
            img (img): The img to display.
        """

        if self._disp_set.norm_enabled:
            try:
                img = self.soft_normalise(img)
            except:
                warnings.warn("Can't normalise image")
                self.set_norm(False)

        if self._disp_set.clahe_enabled:
            try:
                img = self._disp_set.clahe.apply(img)
            except:
                warnings.warn("Can't apply CLAHE filter")
                self.set_clahe(False)

        if self._disp_set.colourmap_enabled:
            try:
                local_mean = cv2.GaussianBlur(img, (25, 25), 5)
                contrast = cv2.subtract(img, local_mean)
                normalised = self.soft_normalise(contrast, 1, 99)
                img = cv2.applyColorMap(normalised, cv2.COLORMAP_INFERNO)
            except:
                warnings.warn("Can't apply colourmap")
                self.set_colourmap(False)
        self.img.setImage(img, axis=0)
        # self.img.setImage(img, axis=0, levels=[0, 255])

    def update_labels(self, lab_data: dict):
        for key, item in lab_data.items():
            self.labs[key].setText(key + ": " + str(item))

    def update_tracking(self, target: Target, timestamp: float, hpr_euler: NDArray):
        uv_path, uv_pt = target.project_from_ned_angles(hpr_euler, timestamp, self._cam_mdl)
        uv_path *= self._scale_factor
        uv_pt *= self._scale_factor


        self.target_crosshairs(uv_pt)


    #     grad = np.gradient(uv_path[:, 1], uv_path[:, 0])
    #     theta = np.arctan(grad)
    #     upper_bound = np.array(
    #         [
    #             uv_path[:, 0] + np.sin(theta) * self.offset,
    #             uv_path[:, 1] - np.cos(theta) * self.offset,
    #         ]
    #     )
    #     lower_bound = np.array(
    #         [
    #             uv_path[:, 0] - np.sin(theta) * self.offset,
    #             uv_path[:, 1] + np.cos(theta) * self.offset,
    #         ]
    #     )
    #     self.traj_bounds_upper.setData(
    #         y=self._display_res[1] - upper_bound[1],
    #         x=1 * upper_bound[0],
    #         pen="b",
    #     )
    #     self.traj_bounds_lower.setData(
    #         y=self._display_res[1] - lower_bound[1],
    #         x=1 * lower_bound[0],
    #         pen="b",
    #     )
    #     self.traj_time_indicator.setData(
    #         y=[
    #             self._display_res[1] - uv_pt[0, 1] + self.offset,
    #             self._display_res[1] - uv_pt[0, 1] + 2 * self.offset,
    #             self._display_res[1] - uv_pt[0, 1] - self.offset,
    #             self._display_res[1] - uv_pt[0, 1] - 2 * self.offset,
    #         ],
    #         x=[
    #             uv_pt[0, 0] - self.offset,
    #             uv_pt[0, 0] - 2 * self.offset,
    #             uv_pt[0, 0] + self.offset,
    #             uv_pt[0, 0] + 2 * self.offset,
    #         ],
    #         connect="pairs",
    #         pen="r",
    #     )

    def target_crosshairs(self, uv_pt: NDArray):
        target_offset = 50
        self.x_target.setData(
            y=[
                self._display_res[1] - uv_pt[0, 1] - target_offset,
                self._display_res[1] - uv_pt[0, 1] + target_offset,
            ],
            x=[
                uv_pt[0, 0],
                uv_pt[0, 0],
            ],
            pen="y",
        )
        self.y_target.setData(
            y=[
                self._display_res[1] - uv_pt[0, 1],
                self._display_res[1] - uv_pt[0, 1],
            ],
            x=[
                uv_pt[0, 0] - target_offset,
                uv_pt[0, 0] + target_offset,
            ],
            pen="y",
        )

    def crosshairs(self):
        self.x_rules.setData(
            y=[
                1 * self._display_res[1] / 8,
                3 * self._display_res[1] / 8,
                5 * self._display_res[1] / 8,
                7 * self._display_res[1] / 8,
            ],
            x=[
                self._display_res[0] / 2,
                self._display_res[0] / 2,
                self._display_res[0] / 2,
                self._display_res[0] / 2,
            ],
            connect="pairs",
            pen="g",
        )
        self.y_rules.setData(
            y=[
                self._display_res[1] / 2,
                self._display_res[1] / 2,
                self._display_res[1] / 2,
                self._display_res[1] / 2,
            ],
            x=[
                1 * self._display_res[0] / 8,
                3 * self._display_res[0] / 8,
                5 * self._display_res[0] / 8,
                7 * self._display_res[0] / 8,
            ],
            connect="pairs",
            pen="g",
        )

    def run(self):
        time.sleep(1)  # Allow some time for everything to start
        try:
            while not self._kill_event.is_set():
                if self._new_stream_event.is_set():
                    self.change_stream()
                frame = self._stream.latest_frame()
                if frame is not None:
                    frame = self._stream.cam.convert_for_monitoring(frame)
                    lab_data = {
                        "Exp": frame.exposure,
                        "Gain": frame.gain,
                        "FPS": np.round(self._stream.cam.frame_rate, 2),
                        "CLAHE": "Enabled" if self._disp_set.clahe_enabled else "",
                        "Saving": (
                            "Enabled" if self._stream.save_enabled else "Disabled"
                        ),
                        "Save queue": self._stream.save_queue_length,
                        "Time": datetime.datetime.fromtimestamp(
                            frame.timestamp, tz=datetime.timezone.utc
                        ).strftime("%H:%M:%S.%f")[:-5],
                    }
                    if self._target is not None:
                        try:
                            hp = self._target.get_head_pitch(frame.timestamp)
                            lab_data["TAR"] = f"Head {hp[0]:.2f} Pitch {hp[1]:.2f}"
                        except Exception as e:
                            warnings.warn(f"Could not get target data: {e}")
                            pass
                    if self._ctx.has_imu_monitor:
                        try:
                            # euler = self._state.extrap_imu_state(frame.timestamp, pc_time=True).hpr
                            euler = self._state.imu_state.hpr
                            lab_data["IMU"] = (
                                f"Head {euler[0]:.2f} Pitch {euler[1]:.2f}"
                            )
                            if self._target is not None:
                                self.update_tracking(self._target, frame.timestamp, np.array(euler))
                        except Exception as e:
                            warnings.warn(f"Could not get IMU data: {e}")
                            try:
                                print(euler)
                            except:
                                pass
                            pass
                    if self._ctx.has_enc_monitor:
                        try:
                            azel = self._state.encoder_state.azel
                            lab_data["GIM"] = f"Az {azel[0]:.2f}, El {azel[1]:.2f}"
                        except:
                            pass

                    img = self.downscale_img(frame.pixels)
                    self.update_img(img)
                    self.update_labels(lab_data)
                    self.app.processEvents()
        finally:
            self.p1.close()
            self.win.close()
            self.app.closeAllWindows()

    def close(self):
        """Close the display window."""
        self._kill_event.set()
