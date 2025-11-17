import time

from obs_cameras.alvium import Alvium811, Alvium508
from obs_cameras.ids import IDSU33080
from obs_cameras.zwo import ASI585
from obs_cameras.base import CameraStream
from obs_display.display import Display
from obs_target.target import PathTarget
from obs_target.sky_test import SkyTarget
from obs_target.parse import read_varda_traj
from obs_controller.controller import GimbalController
from obs_utils.state_plotter import StatePlotter

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
    pt_GS2 = at.Point.from_geodet([-32.150517, 133.689040, 10])
    # pt_Bledisloe = at.Point.from_geodet([-27.511726, 153.0249, 10])

    ids_stream = CameraStream("ids-cam", IDSU33080() , "~/test_cam_data", 50)
    asi_stream = CameraStream("asi-cam", ASI585(), "~/asi_cam_data", 1260)

    # Instantiate state and monitors
    state = State()
    # state_plotter = StatePlotter(state=state, interval=0.5)
    # target = SkyTarget("Canopus", pt_Bledisloe, ids_stream.cam_mdl)
    target = PathTarget(pt_GS2, path_nom, asi_stream.cam_mdl)
    gimbal_controller = GimbalController(target, sink=state.set_gimbal_state)
    gimbal_controller.pi_thread.pc_time(True)
    imu_monitor = CertusMonitor(
        sink=[state.set_imu_state, gimbal_controller.set_imu_state]
    )
    context = Context(streams=[ids_stream, asi_stream], imu_monitor=imu_monitor, controller=gimbal_controller)
    # context = Context(streams=[alv_stream])
    display = Display(context, state, target)
    cli = ObsCLI(context, state, display)

    with context:
        cli.start()
        display.run()

        # gimbal_controller.set_mode("tracking")
        # state_plotter.run()
        # input("Press Enter to continue...")
        # state_plotter.stop()

    # head, pitch = target.get_head_pitch(time.time())
    # print("Head:", head)
    # print("Pitch:", pitch)


if __name__ == "__main__":
    main()
