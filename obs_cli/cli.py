# pyright: standard

import threading
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from obs_cli.cli_autocomplete import (
    initialize_autocompletion,
    add_files_to_autocompletion,
)

from obs_cli.cli_utils import print_centred, progress_print, clean_input
from obs_cli.cli_find_home import find_el_home
from obs_utils.context import Context, State
from obs_display.display import Display


class ObsCLI(threading.Thread):
    _kill_event = threading.Event()
    _ctx: Context
    _state: State
    _display: Display | None = None

    def __init__(self, ctx: Context, state: State, display: Display | None = None):
        super().__init__()
        self._ctx = ctx
        self._state = state
        self._display = display
        self._kill_event = threading.Event()

    def run(self):
        history = InMemoryHistory()
        autocompletion = initialize_autocompletion()
        autocompletion = add_files_to_autocompletion(autocompletion)
        session = PromptSession()

        counter = 0

        camera_id = getattr(self._ctx.disp_stream.cam, "cam_id", None)
        pixel_format = getattr(self._ctx.disp_stream.cam, "pixel_format", None)

        progress_print(f"", "\n")
        print_centred("SCOTI: Spatially Calibrated Optical Tracking and Imaging")
        print_centred("------Real-time Controller-------")
        print_centred("Author: Andrew Lock")
        if camera_id is not None:
            print_centred(f"Camera ID: {camera_id}")
        if pixel_format is not None:
            print_centred(f"Pixel format: {pixel_format}")

        while not self._kill_event.is_set():
            try:
                if self._state.gimbal_state is not None:
                    cur_mode = self._state.gimbal_state.mode
                else:
                    cur_mode = ""
                inp = session.prompt(
                    f"scoti[{counter}] ({cur_mode}) > ",
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=autocompletion,
                )

                # Change mode
                if inp.startswith("mode "):
                    mode = inp.split(" ")[1]
                    if mode in ["manual", "tracking"]:
                        self._ctx.controller.set_mode(mode)
                        print(f"Switched to {mode} mode")
                    else:
                        print(f"Unknown mode: {mode}")

                        print("Available modes: manual, tracking")

                # Trigger save
                if inp in ["start save", "save"]:
                    self._ctx.save_all()

                if inp == "stop save":
                    self._ctx.stop_saving_all()

                # SIngle key exposure and gain
                if inp == "E":
                    exp = self._ctx.disp_stream.cam.exposure
                    self._ctx.disp_stream.cam.set_exposure(int(exp * 1.2))

                if inp == "e":
                    exp = self._ctx.disp_stream.cam.exposure
                    self._ctx.disp_stream.cam.set_exposure(int(exp * 0.8))

                if inp == "G":
                    gain = self._ctx.disp_stream.cam.gain
                    self._ctx.disp_stream.cam.set_gain(int(gain * 1.2))

                if inp == "g":
                    gain = self._ctx.disp_stream.cam.gain
                    self._ctx.disp_stream.cam.set_exposure(int(gain * 0.8))

                # Change camera stream
                if inp.startswith("view"):
                    line = inp.split(" ")
                    if line[1].isdigit():
                        self._ctx.change_display_stream(int(line[1]))
                        self._display.signal_new_stream()
                    else:
                        print("Could not change streams. Enter digit of stream")

                # Set parameters
                if inp.startswith("set "):
                    line = inp.split(" ")
                    if line[1] == "e":
                        exp = line[2]
                        self._ctx.disp_stream.cam.set_exposure(int(exp))

                    elif line[1] == "g":
                        gain = line[2]
                        self._ctx.disp_stream.cam.set_gain(int(gain))


                    elif line[1] == "az":
                        if line[2] == "home":
                            self._ctx.controller.set_home("az")
                        if line[2] == "ulim":
                            self._ctx.controller.set_limit("az", "max")
                        if line[2] == "llim":
                            self._ctx.controller.set_limit("az", "min")

                    elif line[1] == "el":
                        if line[2] == "home":
                            self._ctx.controller.set_home("el")
                        if line[2] == "ulim":
                            self._ctx.controller.set_limit("el", "max")
                        if line[2] == "llim":
                            self._ctx.controller.set_limit("el", "min")

                    if self._display is not None:
                        if line[1] == "clahe":
                            if line[2] == "on":
                                try:
                                    clip_limit = float(line[3])
                                except:
                                    clip_limit = 2.0
                                self._display.set_clahe(True, clip_limit)
                            elif line[2] == "off":
                                self._display.set_clahe(False)

                        elif line[1] == "norm":
                            if line[2] == "on":
                                self._display.set_norm(True)
                            elif line[2] == "off":
                                self._display.set_norm(False)

                        elif line[1] == "colour":
                            if line[2] == "on":
                                self._display.set_colourmap(True)
                            elif line[2] == "off":
                                self._display.set_colourmap(False)

                        elif line[1] == "hist":
                            if line[2] == "on":
                                self._display.set_hist_location("bottom")
                            elif line[2] in ["bottom", "right", "off"]:
                                self._display.set_hist_location(line[2])

                        elif line[1] == "sat":
                            if line[2] == "on":
                                self._display.set_saturation_overlay(True)
                            elif line[2] == "off":
                                self._display.set_saturation_overlay(False)

                if inp.startswith("reset "):
                    line = inp.split(" ")
                    if line[1] == "limits":
                        self._ctx.controller.reset_limits()

                # if inp.startswith("SP"):
                #     line = inp.split(" ")
                #     try:
                #         self._ctx.controller.set_setpoint(float(line[1]), float(line[2]))
                #     except:
                #         print("Error setting setpoint")
                #
                # if inp == "find el home":
                #     find_el_home(self._ctx)

                # if inp == "starcal":
                #     starcal(self._ctx)
                #     autocompletion = add_files_to_autocompletion(autocompletion)

                # if inp.startswith("load "):
                #     line = inp.split(" ")
                #     if line[1] == "orientation":
                #         filepath = line[2]
                #         load_orientation(self._ctx, filepath)
                #     elif line[1] == "fov":
                #         filepath = line[2]
                #         load_fov(self._ctx, filepath)

                # if inp.startswith("target "):
                #     line = inp.split(" ")
                #     body = line[1]
                #     target_planet(self._ctx, body)

                # if inp.startswith("SC"):
                #     line = inp.split(" ")
                #     az, el = self._ctx.gimbal_frame.az_el_from_head_el(float(line[1]), float(line[2]))
                #     az = float(((az + 180) % 360) - 180)
                #     el = float(((el + 180) % 360) - 180)
                #     to_setpoint(self._ctx.controller, az, el)


                if inp.startswith("exit") or inp.startswith("quit"):
                    if self._display is not None:
                        self._display.close()
                    self._kill_event.set()
                    print("\nExiting SCOTI")

                if self._display is not None:
                    if self._display._kill_event.is_set():
                        print("\nDisplay closed. Exiting SCOTI")
                        self._kill_event.set()

                counter += 1

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt: Exiting REPL")
                if self._display is not None:
                    self._display.close()
                self._kill_event.set()

            except Exception as e:
                print("    ERROR PROCESSING COMMAND")
                print(e)
                print("------------")

