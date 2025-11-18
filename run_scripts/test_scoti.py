from obs_cameras.ids import IDSU33080
from obs_cameras.zwo import ASI585
from obs_cameras.base import CameraStream
from obs_display.display import Display
# from obs_target.target import PathTarget
from obs_target.sky_target import SkyTarget
# from obs_target.parse import read_varda_traj
from obs_certus.monitor import CertusMonitor
# from obs_encoders.monitor import EncoderMonitor
from obs_controller.controller import GimbalController
from obs_utils.context import Context, State
from obs_cli.cli import ObsCLI


import astrix as at


def main():
    # Create objects from data
    pt_home = at.Point.from_geodet([-27.511644, 153.0245, 30])

    alv_stream = CameraStream("alv-cam", IDSU33080(), "~/test_cam_data", 50)
    # zwo_stream = CameraStream("asi-cam", ASI585(), "~/asi_cam_data", 1260)

    # Instantiate state and monitors
    state = State()
    target = SkyTarget("Aldebaran", pt_home)
    # target = SkyTarget("Sirius", pt_home)
    controller = GimbalController(target, sink=state.set_gimbal_state)
    imu_monitor = CertusMonitor(sink=[state.set_imu_state, controller.set_imu_state])
    controller.pi_thread.pc_time(True)
    context = Context(
        streams=[alv_stream],
        imu_monitor=imu_monitor,
        controller=controller,
    )
    display = Display(context, state, target)
    cli = ObsCLI(context, state, display)

    with context:
        cli.start()
        display.run()

if __name__ == "__main__":
    main()
 
