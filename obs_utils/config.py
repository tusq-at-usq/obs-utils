from pathlib import Path

import yaml


CAMERA_CONFIG_DEFAULTS: dict = {
    "camera_id": None,
    "save_root_dir": "~/test_cam_data",
    "focal_length_mm": 50,
    "pixel_format": "Mono8",
    "sensor_bit_depth": None,
    "monitor_rotation_deg": 90,
    "monitor_flip_x": False,
    "monitor_flip_y": False,
    "binning_factor": 1,
    "binning_mode": None,
    "startup_exposure": 20,
    "startup_gain": 1,
}


def load_camera_config(config_path: Path) -> dict:
    """Load camera config YAML and merge with default settings."""
    if not config_path.exists():
        return CAMERA_CONFIG_DEFAULTS.copy()

    with open(config_path, "r") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    return {**CAMERA_CONFIG_DEFAULTS, **loaded}


def apply_camera_settings(camera, settings: dict) -> None:
    """Apply standard config fields to any camera interface object.

    This is the single place where config keys are mapped to camera
    attributes, so every mission script benefits automatically.
    """
    camera.cam_id = settings.get("camera_id")
    camera.pixel_format = settings["pixel_format"]
    camera.sensor_bit_depth = settings["sensor_bit_depth"]
    camera.monitor_rotation_deg = int(settings.get("monitor_rotation_deg", 90) or 0)
    camera.monitor_flip_x = bool(settings.get("monitor_flip_x", False))
    camera.monitor_flip_y = bool(settings.get("monitor_flip_y", False))
    camera.EXP_DEFAULT = settings["startup_exposure"]
    camera.GAIN_DEFAULT = settings["startup_gain"]

    # Binning — binning_factor sets both axes; individual overrides take precedence.
    binning_factor = int(settings.get("binning_factor", 1) or 1)
    camera.binning_horizontal = int(
        settings.get("binning_horizontal", binning_factor) or binning_factor
    )
    camera.binning_vertical = int(
        settings.get("binning_vertical", binning_factor) or binning_factor
    )
    camera.binning_mode = settings.get("binning_mode")
