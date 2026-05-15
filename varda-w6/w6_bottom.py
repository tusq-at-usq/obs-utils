from obs_cameras.alvium import AlviumU130VSWIR
from obs_cameras.base import CameraStream
from obs_display.display import Display
from obs_target.zafiro_azel_target import ZafiroSystemsAzElTarget

# from obs_certus.monitor import CertusMonitor
from obs_encoders.monitor import EncoderMonitor
from obs_utils.config import load_camera_config
from obs_utils.context import Context, State
from obs_cli.cli import ObsCLI

from pathlib import Path
from typing import Literal

import astrix as at


OverlayMode = Literal["SIMULATE_REENTRY", "CSV_ABSOLUTE_TIME"]

SCRIPT_CAMERA_CONFIG = "swir_bottom.yaml"
SCRIPT_ENCODER_CONFIG = "encoders_config.yaml"
SCRIPT_AZEL_CSV = "varda-w6/20260512_GS_Az-El_data.csv"
DEFAULT_OVERLAY_MODE: OverlayMode = "SIMULATE_REENTRY"
DEFAULT_REENTRY_DELAY_SECONDS = 15.0
DEFAULT_USE_CSV_TIME_OF_DAY_ONLY = True


def _build_paths(script_dir: Path) -> tuple[Path, Path, Path]:
    camera_config = script_dir / SCRIPT_CAMERA_CONFIG
    encoder_config = script_dir / SCRIPT_ENCODER_CONFIG
    azel_csv_path = script_dir.parent / SCRIPT_AZEL_CSV
    return camera_config, encoder_config, azel_csv_path


def _build_ground_station_point() -> at.Point:
    # does not do anything for W6 as not using ECEF mode, but required for running. Putting in nominal location for records
    return at.Point.from_geodet([-31.82, 132.48, 10])


def _build_camera_stream(camera_settings: dict) -> CameraStream:
    camera = AlviumU130VSWIR()
    camera.cam_id = camera_settings["camera_id"]
    camera.pixel_format = camera_settings["pixel_format"]
    camera.sensor_bit_depth = camera_settings["sensor_bit_depth"]
    camera.EXP_DEFAULT = camera_settings["startup_exposure"]
    camera.GAIN_DEFAULT = camera_settings["startup_gain"]

    return CameraStream(
        "alv-cam",
        camera,
        camera_settings["save_root_dir"],
        camera_settings["focal_length_mm"],
    )


def _build_target(
    ground_point: at.Point,
    azel_csv_path: Path,
    overlay_mode: OverlayMode,
    reentry_delay_seconds: float,
) -> ZafiroSystemsAzElTarget:
    return ZafiroSystemsAzElTarget(
        ground_point,
        str(azel_csv_path),
        simulate_reentry=(overlay_mode == "SIMULATE_REENTRY"),
        start_delay_seconds=reentry_delay_seconds,
        use_csv_time_of_day_only=DEFAULT_USE_CSV_TIME_OF_DAY_ONLY,
    )


def _build_runtime(
    stream: CameraStream,
    encoder_config: Path,
    target: ZafiroSystemsAzElTarget,
) -> tuple[Context, State, Display, ObsCLI]:
    state = State()
    enc_monitor = EncoderMonitor(
        config_filepath=encoder_config,
        sink=[state.set_encoder_state],
    )
    context = Context(
        streams=[stream],
        enc_monitor=enc_monitor,
    )
    display = Display(context, state, target)
    cli = ObsCLI(context, state, display)
    return context, state, display, cli


def main():
    script_dir = Path(__file__).resolve().parent
    camera_config, encoder_config, azel_csv_path = _build_paths(script_dir)
    overlay_mode: OverlayMode = DEFAULT_OVERLAY_MODE
    reentry_delay_seconds = DEFAULT_REENTRY_DELAY_SECONDS

    camera_settings = load_camera_config(camera_config)
    ground_point = _build_ground_station_point()
    stream = _build_camera_stream(camera_settings)
    target = _build_target(
        ground_point,
        azel_csv_path,
        overlay_mode,
        reentry_delay_seconds,
    )
    context, _, display, cli = _build_runtime(stream, encoder_config, target)

    with context:
        cli.start()
        display.run()

if __name__ == "__main__":
    main()
