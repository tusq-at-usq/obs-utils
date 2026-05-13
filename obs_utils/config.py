from pathlib import Path

import yaml


CAMERA_CONFIG_DEFAULTS: dict = {
    "camera_id": None,
    "save_root_dir": "~/test_cam_data",
    "focal_length_mm": 50,
    "pixel_format": "Mono8",
    "sensor_bit_depth": None,
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
