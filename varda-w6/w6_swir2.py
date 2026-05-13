from obs_cameras.alvium import AlviumU130VSWIR
from obs_cameras.base import CameraStream
from obs_display.display import Display
from obs_target.target import AzElCsvTarget, PathTarget
from obs_target.parse import read_varda_traj

# from obs_certus.monitor import CertusMonitor
from obs_encoders.monitor import EncoderMonitor
from obs_utils.config import load_camera_config
from obs_utils.context import Context, State
from obs_cli.cli import ObsCLI

from pathlib import Path

import astrix as at


def main():
    script_dir = Path(__file__).resolve().parent
    camera_config = script_dir / "swir_2.yaml"
    encoder_config = script_dir / "encoders_config.yaml"
    overlay_target_mode = "ecef"  # "ecef" or "azel_csv"
    azel_csv_path = Path("~/obs-utils/20260512_GS_Az-El_data.csv").expanduser()
    # imu_config = script_dir / "certus_config.yaml"
    camera_settings = load_camera_config(camera_config)

    # Create objects from data
    pt_GS1 = at.Point.from_geodet([-31.988851, 132.437920, 10])

    if overlay_target_mode == "azel_csv":
        target = AzElCsvTarget(pt_GS1, str(azel_csv_path))
    else:
        path_nom = read_varda_traj(
            "~/varda-w4/planning/data/W4_Nominal_ECEF.csv",
            test_time_adjustment=True,
        )
        target = PathTarget(pt_GS1, path_nom)

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
