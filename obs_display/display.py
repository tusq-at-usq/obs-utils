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
    hist_location: str = "bottom"
    saturation_overlay_enabled: bool = True
    text_overlay_location: str = "right"


class Display:
    _kill_event: threading.Event
    _new_stream_event: threading.Event
    _disp_lock: threading.RLock

    app: PyQt6.QtWidgets.QApplication
    win: pg.GraphicsLayoutWidget
    p1: pg.ViewBox
    hist: pg.HistogramLUTItem | None
    img: pg.ImageItem
    sat_overlay: pg.ImageItem
    labs: dict
    x_rules: pg.PlotDataItem
    y_rules: pg.PlotDataItem
    _scale_factor: float
    _display_res: tuple[int, int]
    _width: int
    _sidebar_width: int
    _footer_height: int

    _disp_set: DisplaySettings

    _HINTS: dict = {
        "Exp":        "e/E",
        "Gain":       "g/G",
        "FPS":        "",
        "CLAHE":      "c",
        "SAT":        "r",
        "Saving":     "s",
        "Save queue": "",
        "GIM":        "",
        "IMU":        "",
        "TAR":        "",
        "TrackAlt":   "",
        "Hist":       "h",
        "Layout":     "l",
    }

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
        self._sidebar_width = 420
        self._footer_height = 120

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
        sat_overlay = pg.ImageItem()
        exp_lab = pg.TextItem()
        gain_lab = pg.TextItem()
        fps_lab = pg.TextItem()
        clahe_lab = pg.TextItem()
        sat_lab = pg.TextItem()
        saving_lab = pg.TextItem()
        save_queue = pg.TextItem()
        gimb_angles_lab = pg.TextItem()
        target_angles_lab = pg.TextItem()
        imu_angles_lab = pg.TextItem()
        orient_cal_lab = pg.TextItem()
        track_alt_lab = pg.TextItem()
        hist_lab = pg.TextItem()
        layout_lab = pg.TextItem()

        self.p1.addItem(img)
        self.p1.addItem(sat_overlay)
        self.p1.addItem(exp_lab)
        self.p1.addItem(gain_lab)
        self.p1.addItem(fps_lab)
        self.p1.addItem(clahe_lab)
        self.p1.addItem(sat_lab)
        self.p1.addItem(saving_lab)
        self.p1.addItem(save_queue)
        self.p1.addItem(gimb_angles_lab)
        self.p1.addItem(target_angles_lab)
        self.p1.addItem(imu_angles_lab)
        self.p1.addItem(orient_cal_lab)
        self.p1.addItem(track_alt_lab)
        self.p1.addItem(hist_lab)
        self.p1.addItem(layout_lab)

        self.img = img
        self.sat_overlay = sat_overlay
        self.sat_overlay.setZValue(10)
        self.hist = None
        self.set_hist_location(self._disp_set.hist_location)
        self.labs = {
            "Exp": exp_lab,
            "Gain": gain_lab,
            "FPS": fps_lab,
            "CLAHE": clahe_lab,
            "SAT": sat_lab,
            "Saving": saving_lab,
            "Save queue": save_queue,
            "GIM": gimb_angles_lab,
            "IMU": imu_angles_lab,
            "TAR": target_angles_lab,
            "TrackAlt": track_alt_lab,
            "Hist": hist_lab,
            "Layout": layout_lab,
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
            new_exp = max(old_exp * 1.1, old_exp + 0.1)
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
        elif txt in ["H", "h"]:
            self.cycle_hist_location()
        elif txt in ["R", "r"]:
            self.set_saturation_overlay(not self._disp_set.saturation_overlay_enabled)
        elif txt in ["L", "l"]:
            self.cycle_text_overlay_location()
        elif txt in ["Q", "q"]:
            self.close()

    def set_saturation_overlay(self, enabled: bool):
        with self._disp_lock:
            self._disp_set.saturation_overlay_enabled = enabled
            if not enabled:
                self.sat_overlay.clear()

    def set_hist_location(self, location: str):
        location = location.lower()
        if location not in {"bottom", "right", "off"}:
            raise ValueError("Histogram location must be one of: bottom, right, off")

        if self.hist is not None:
            self.win.removeItem(self.hist)
            self.hist = None

        if location != "off":
            orientation = "horizontal" if location == "bottom" else "vertical"
            self.hist = pg.HistogramLUTItem(self.img, orientation=orientation)
            if location == "bottom":
                self.win.addItem(self.hist, 1, 0)
            else:
                self.win.addItem(self.hist, 0, 1)

        self._disp_set.hist_location = location

    def cycle_hist_location(self):
        current = self._disp_set.hist_location
        next_location = {
            "bottom": "right",
            "right": "off",
            "off": "bottom",
        }[current]
        self.set_hist_location(next_location)

    def set_text_overlay_location(self, location: str):
        location = location.lower()
        if location not in {"right", "bottom", "overlay"}:
            raise ValueError("Text overlay location must be one of: right, bottom, overlay")

        with self._disp_lock:
            self._disp_set.text_overlay_location = location
            self.set_bounds()
            self._position_text_labels()
            self.app.processEvents()

    def cycle_text_overlay_location(self):
        current = self._disp_set.text_overlay_location
        next_location = {
            "right": "bottom",
            "bottom": "overlay",
            "overlay": "right",
        }[current]
        self.set_text_overlay_location(next_location)

    def set_bounds(self):
        # self.img_size = self._stream.cam.frame_res
        aspect = self._stream.cam.frame_res[0] / self._stream.cam.frame_res[1]
        height = self._width / aspect
        self._display_res = (int(self._width), int(height))
        self._scale_factor = self._display_res[0] / self._stream.cam.frame_res[0]

        # Set constant bounds for the view
        if self._disp_set.text_overlay_location == "right":
            x_range = (0, self._display_res[0] + self._sidebar_width)
            y_range = (0, self._display_res[1])
        elif self._disp_set.text_overlay_location == "bottom":
            x_range = (0, self._display_res[0])
            y_range = (0, self._display_res[1] + self._footer_height)
        else:  # overlay
            x_range = (0, self._display_res[0])
            y_range = (0, self._display_res[1])

        self.p1.setRange(
            xRange=x_range,
            yRange=y_range,
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

    def downscale_mask(self, mask: np.ndarray) -> np.ndarray:
        if self._scale_factor != 1.0:
            new_size = (
                int(mask.shape[1] * self._scale_factor),
                int(mask.shape[0] * self._scale_factor),
            )
            resized = cv2.resize(
                mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST
            )
            return resized.astype(bool)
        return mask.astype(bool)

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
        if self._disp_set.text_overlay_location == "right":
            win_dims = (
                int(self._width + self._sidebar_width),
                int(self._width * (1 / img_ratio)),
            )
        elif self._disp_set.text_overlay_location == "bottom":
            win_dims = (
                int(self._width),
                int(self._width * (1 / img_ratio) + self._footer_height),
            )
        else:  # overlay
            win_dims = (
                int(self._width),
                int(self._width * (1 / img_ratio)),
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

        self._set_text_label_styles()
        self._position_text_labels()

        self.crosshairs()

        self.app.processEvents()
        # self.win.showMaximized()
        self.win.setContentsMargins(0, 0, 0, 0)
        self.win.show()

    def _set_text_label_styles(self):
        is_bottom = self._disp_set.text_overlay_location == "bottom"

        self.labs["FPS"].setFont(QtGui.QFont("monospace", 11, 150))
        self.labs["FPS"].setColor("cyan")

        self.labs["CLAHE"].setFont(QtGui.QFont("monospace", 10, 150))
        self.labs["CLAHE"].setColor("cyan")

        self.labs["SAT"].setFont(QtGui.QFont("monospace", 10, 150))
        self.labs["SAT"].setColor("cyan")

        self.labs["Gain"].setFont(QtGui.QFont("monospace", 11, 150))
        self.labs["Gain"].setColor("cyan")

        self.labs["Exp"].setFont(QtGui.QFont("monospace", 11, 150))
        self.labs["Exp"].setColor("cyan")

        saving_size = 15 if is_bottom else 13
        self.labs["Saving"].setFont(QtGui.QFont("monospace", saving_size, 700))
        self.labs["Saving"].setColor("r")

        self.labs["Save queue"].setFont(QtGui.QFont("monospace", 10, 700))
        self.labs["Save queue"].setColor("r")

        gim_size = 14 if is_bottom else 12
        self.labs["GIM"].setFont(QtGui.QFont("monospace", gim_size, 700))
        self.labs["GIM"].setColor("g")

        self.labs["IMU"].setFont(QtGui.QFont("monospace", 11, 500))
        self.labs["IMU"].setColor("g")

        tar_size = 14 if is_bottom else 12
        self.labs["TAR"].setFont(QtGui.QFont("monospace", tar_size, 700))
        self.labs["TAR"].setColor("y")

        self.labs["TrackAlt"].setFont(QtGui.QFont("monospace", 10, 150))
        self.labs["TrackAlt"].setColor("g")

        self.labs["Hist"].setFont(QtGui.QFont("monospace", 10, 150))
        self.labs["Hist"].setColor("cyan")
        self.labs["Hist"].setText("Hist  [h]")

        self.labs["Layout"].setFont(QtGui.QFont("monospace", 10, 150))
        self.labs["Layout"].setColor("cyan")
        self.labs["Layout"].setText("Layout  [l]")

    def _position_text_labels(self):
        ys = self._display_res[1] / 1080  # y scale factor
        xs = self._display_res[0] / 1920  # x scale factor

        if self._disp_set.text_overlay_location == "overlay":
            # Top-left: camera controls
            self.labs["FPS"].setPos(20 * xs,   self._display_res[1] - 30 * ys)
            self.labs["Gain"].setPos(20 * xs,  self._display_res[1] - 55 * ys)
            self.labs["Exp"].setPos(20 * xs,   self._display_res[1] - 80 * ys)
            self.labs["CLAHE"].setPos(20 * xs, self._display_res[1] - 105 * ys)
            self.labs["SAT"].setPos(20 * xs,   self._display_res[1] - 130 * ys)
            # Top-right: recording
            self.labs["Saving"].setPos(self._display_res[0] - 200 * xs, self._display_res[1] - 30 * ys)
            self.labs["Save queue"].setPos(self._display_res[0] - 200 * xs, self._display_res[1] - 55 * ys)
            # Bottom-left: pointing
            self.labs["TAR"].setPos(20 * xs,       80 * ys)
            self.labs["GIM"].setPos(20 * xs,       55 * ys)
            self.labs["IMU"].setPos(20 * xs,       30 * ys)
            # Bottom-right: misc
            self.labs["TrackAlt"].setPos(self._display_res[0] - 200 * xs, 80 * ys)
            self.labs["Hist"].setPos(self._display_res[0] - 200 * xs,    55 * ys)
            self.labs["Layout"].setPos(self._display_res[0] - 200 * xs,  30 * ys)
            return

        if self._disp_set.text_overlay_location == "right":
            x = self._display_res[0] + 20
            # Groups separated by 15px gap within, 30px between groups
            y_map = {
                # --- Recording ---
                "Saving":     30,
                "Save queue": 55,
                # --- Pointing ---
                "TAR":        100,
                "GIM":        125,
                "IMU":        150,
                # --- Camera ---
                "Exp":        195,
                "Gain":       220,
                "FPS":        245,
                # --- Display ---
                "CLAHE":      290,
                "SAT":        315,
                "TrackAlt":   340,
                # --- Controls ---
                "Hist":       385,
                "Layout":     410,
            }
            for key, y in y_map.items():
                self.labs[key].setPos(x, y * ys)
            return

        xs = self._display_res[0] / 1920   # x scale factor
        y1 = self._display_res[1] + 72     # primary row   (visually top of strip)
        y2 = self._display_res[1] + 42     # secondary row
        y3 = self._display_res[1] + 12     # tertiary row  (visually bottom of strip)

        # --- group 1: Recording (left) ----------------------
        self.labs["Saving"].setPos(20 * xs, y1)
        self.labs["Save queue"].setPos(20 * xs, y2)
        self.labs["FPS"].setPos(20 * xs, y3)

        # --- group 2: Pointing (left-centre) ----------------
        self.labs["TAR"].setPos(380 * xs, y1)
        self.labs["GIM"].setPos(380 * xs, y2)
        self.labs["IMU"].setPos(730 * xs, y2)

        # --- group 3: Camera (right-centre) -----------------
        self.labs["Exp"].setPos(1060 * xs, y1)
        self.labs["Gain"].setPos(1060 * xs, y2)

        # --- group 4: Display (right) -----------------------
        self.labs["CLAHE"].setPos(1480 * xs, y1)
        self.labs["SAT"].setPos(1480 * xs, y2)
        self.labs["TrackAlt"].setPos(1680 * xs, y2)

        # --- group 5: Controls (far right) ------------------
        self.labs["Hist"].setPos(1750 * xs, y1)
        self.labs["Layout"].setPos(1750 * xs, y2)

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

    def _resolve_sensor_bit_depth(self, img: np.ndarray) -> int | None:
        sensor_bit_depth = getattr(self._stream.cam, "sensor_bit_depth", None)
        pixel_format = getattr(self._stream.cam, "pixel_format", "")

        if isinstance(sensor_bit_depth, str) and sensor_bit_depth.startswith("Bpp"):
            try:
                return int(sensor_bit_depth[3:])
            except ValueError:
                pass
        if isinstance(sensor_bit_depth, int):
            return sensor_bit_depth

        if isinstance(pixel_format, str):
            digits = "".join(ch for ch in pixel_format if ch.isdigit())
            if digits:
                return int(digits)

        if np.issubdtype(img.dtype, np.integer):
            return int(np.iinfo(img.dtype).bits)
        return None

    def get_saturation_limits(self, img: np.ndarray) -> tuple[int, int] | None:
        if not np.issubdtype(img.dtype, np.integer):
            return None
        bit_depth = self._resolve_sensor_bit_depth(img)
        if bit_depth is None:
            return None

        dtype_bits = int(np.iinfo(img.dtype).bits)
        full_scale = (1 << bit_depth) - 1
        right_aligned_limit = full_scale
        if bit_depth < dtype_bits:
            left_shift = dtype_bits - bit_depth
            left_aligned_limit = full_scale << left_shift
        else:
            left_aligned_limit = full_scale
        return right_aligned_limit, left_aligned_limit

    def compute_saturation_mask(self, img: np.ndarray) -> np.ndarray | None:
        if not np.issubdtype(img.dtype, np.integer):
            return None

        sat_limits = self.get_saturation_limits(img)
        if sat_limits is None:
            return None

        right_limit, left_limit = sat_limits
        threshold_pad = 1

        if img.ndim == 2:
            sat_right = img >= max(0, right_limit - threshold_pad)
            sat_left = img >= max(0, left_limit - threshold_pad)
            return sat_right | sat_left
        if img.ndim == 3:
            sat_right = np.any(img >= max(0, right_limit - threshold_pad), axis=2)
            sat_left = np.any(img >= max(0, left_limit - threshold_pad), axis=2)
            return sat_right | sat_left
        return None

    def _update_saturation_overlay(self, saturation_mask: np.ndarray | None = None):
        if not self._disp_set.saturation_overlay_enabled or saturation_mask is None:
            self.sat_overlay.clear()
            return

        if saturation_mask.ndim != 2:
            self.sat_overlay.clear()
            return

        overlay = np.zeros((saturation_mask.shape[0], saturation_mask.shape[1], 4), dtype=np.uint8)
        overlay[saturation_mask, 0] = 255
        overlay[saturation_mask, 3] = 180
        self.sat_overlay.setImage(overlay, axis=0)

    def update_img(self, img, saturation_mask: np.ndarray | None = None):
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
        try:
            sat_mask = None if saturation_mask is None else saturation_mask.astype(bool)
            self._update_saturation_overlay(sat_mask)
        except Exception as e:
            warnings.warn(f"Can't apply saturation overlay: {e}")
            self.sat_overlay.clear()
        # self.img.setImage(img, axis=0, levels=[0, 255])

    _STATIC_LABS = {"Hist", "Layout"}

    def update_labels(self, lab_data: dict):
        for key, item in lab_data.items():
            if key in self._STATIC_LABS:
                continue
            hint = self._HINTS.get(key, "")
            suffix = f"  [{hint}]" if hint else ""
            self.labs[key].setText(key + ": " + str(item) + suffix)

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
                    raw_pixels = frame.pixels
                    raw_sat_mask = self.compute_saturation_mask(raw_pixels)

                    frame = self._stream.cam.convert_for_monitoring(frame)
                    lab_data = {
                        "Exp": f"{frame.exposure:.1f}",
                        "Gain": f"{frame.gain:.1f}",
                        "FPS": f"{self._stream.cam.frame_rate:.1f}",
                        "CLAHE": "Enabled" if self._disp_set.clahe_enabled else "",
                        "SAT": "Enabled" if self._disp_set.saturation_overlay_enabled else "",
                        "Saving": (
                            "REC" if self._stream.save_enabled else "IDLE"
                        ),
                        "Save queue": self._stream.save_queue_length,
                    }
                    self.labs["Saving"].setColor(
                        "g" if self._stream.save_enabled else "r"
                    )
                    if self._target is not None:
                        try:
                            hp = self._target.get_head_pitch(frame.timestamp)
                            lab_data["TAR"] = f"Az {hp[0]:>7.1f}  El {hp[1]:>6.1f}"
                        except Exception as e:
                            warnings.warn(f"Could not get target data: {e}")
                            pass
                    if self._ctx.has_imu_monitor:
                        try:
                            # euler = self._state.extrap_imu_state(frame.timestamp, pc_time=True).hpr
                            euler = self._state.imu_state.hpr
                            lab_data["IMU"] = (
                                f"Az {euler[0]:>7.1f}  El {euler[1]:>6.1f}"
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
                            lab_data["GIM"] = f"Az {azel[0]:>7.1f}  El {azel[1]:>6.1f}"
                        except:
                            pass

                    img = self.downscale_img(frame.pixels)
                    saturation_mask = raw_sat_mask
                    if saturation_mask is not None:
                        saturation_mask = self._stream.cam.convert_mask_for_monitoring(saturation_mask)
                    if saturation_mask is not None:
                        saturation_mask = self.downscale_mask(saturation_mask)
                    self.update_img(img, saturation_mask=saturation_mask)
                    self.update_labels(lab_data)
                    self.app.processEvents()
        finally:
            self.p1.close()
            self.win.close()
            self.app.closeAllWindows()

    def close(self):
        """Close the display window."""
        self._kill_event.set()
