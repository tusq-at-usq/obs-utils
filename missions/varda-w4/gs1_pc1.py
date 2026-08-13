from obs_cameras.alvium import Alvium811, Alvium508
from obs_cameras.zwo import ASI585
from obs_cameras.base import CameraStream
from obs_display.display import Display
from obs_target.target import PathTarget
from obs_target.parse import read_varda_traj

from obs_certus.monitor import CertusMonitor
from obs_encoders.monitor import EncoderMonitor
from obs_utils.context import Context, State
from obs_cli.cli import ObsCLI


import astrix as at


def main():
    # Create objects from data
    path_nom = read_varda_traj(
        "~/varda-w4/planning/data/W4_Nominal_ECEF.csv",
        test_time_adjustment=True,
    )
    pt_GS1 = at.Point.from_geodet([-31.988851, 132.437920, 10])

    alv_stream = CameraStream("alv-cam", Alvium811(), "~/test_cam_data", 50)
    # zwo_stream = CameraStream("asi-cam", ASI585(), "~/asi_cam_data", 1260)

    # Instantiate state and monitors
    state = State()
    target = PathTarget(pt_GS1, path_nom)
    enc_monitor = EncoderMonitor(sink=[state.set_encoder_state])
    imu_monitor = CertusMonitor(sink=[state.set_imu_state])
    context = Context(
        streams=[alv_stream],
        imu_monitor=imu_monitor,
        enc_monitor=enc_monitor,
    )
    display = Display(context, state, target)
    cli = ObsCLI(context, state, display)

    with context:
        cli.start()
        display.run()

if __name__ == "__main__":
    main()
