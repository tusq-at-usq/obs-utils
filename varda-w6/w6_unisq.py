from obs_cameras.alvium import AlviumU130VSWIR
from obs_cameras.base import CameraStream
from obs_display.display import Display
from obs_target.target import PathTarget
from obs_target.parse import read_varda_traj

# from obs_certus.monitor import CertusMonitor
from obs_encoders.monitor import EncoderMonitor
from obs_utils.context import Context, State
from obs_cli.cli import ObsCLI

from pathlib import Path
import yaml

import astrix as at


def load_camera_config(config_path: Path) -> dict:
    defaults = {
        "camera_id": None,
        "save_root_dir": "~/test_cam_data",
        "focal_length_mm": 50,
        "pixel_format": "Mono8",
        "sensor_bit_depth": None,
        "startup_exposure": 20,
        "startup_gain": 1,
    }

    if not config_path.exists():
        return defaults

    with open(config_path, "r") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    return {**defaults, **loaded}


def main():
    script_dir = Path(__file__).resolve().parent
    camera_config = script_dir / "swir_1312.yaml"
    encoder_config = script_dir / "encoders_config.yaml"
    # imu_config = script_dir / "certus_config.yaml"
    camera_settings = load_camera_config(camera_config)

    # Create objects from data
    path_nom = read_varda_traj(
        "~/varda-w4/planning/data/W4_Nominal_ECEF.csv",
        test_time_adjustment=True,
    )
    pt_GS1 = at.Point.from_geodet([-31.988851, 132.437920, 10])

    camera = AlviumU130VSWIR()
    camera.cam_id = camera_settings["camera_id"]
    camera.pixel_format = camera_settings["pixel_format"]
    camera.sensor_bit_depth = camera_settings["sensor_bit_depth"]
    camera.EXP_DEFAULT = camera_settings["startup_exposure"]
    camera.GAIN_DEFAULT = camera_settings["startup_gain"]

    alv_stream = CameraStream(
        "alv-cam",
        camera,
        camera_settings["save_root_dir"],
        camera_settings["focal_length_mm"],
    )
    # zwo_stream = CameraStream("asi-cam", ASI585(), "~/asi_cam_data", 1260)

    # Instantiate state and monitors
    state = State()
    target = PathTarget(pt_GS1, path_nom)
    enc_monitor = EncoderMonitor(
        config_filepath=encoder_config,
        sink=[state.set_encoder_state],
    )
    # imu_monitor = CertusMonitor(config_filepath=imu_config, sink=[state.set_imu_state])
    context = Context(
        streams=[alv_stream],
        # imu_monitor=imu_monitor,
        enc_monitor=enc_monitor,
    )
    display = Display(context, state, target)
    cli = ObsCLI(context, state, display)

    with context:
        cli.start()
        display.run()

if __name__ == "__main__":
    main()
