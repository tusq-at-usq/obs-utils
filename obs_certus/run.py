import os
import yaml
import zmq
import subprocess
import time
import select
import json
from rich.table import Table
from rich.live import Live
from rich.console import Console
from importlib.resources import files

from obs_utils.discovery import port_serial_search

CERTUS_EXEC = str(files("obs_certus").joinpath("cpp", "certus", "stream_packets"))

# CERTUS_EXEC = os.path.join(
#     os.path.dirname(__file__), "./", "cpp", "certus", "stream_packets"
# )

class IMUDisplay:
    def __init__(self, config: dict) -> None:
        self.console = Console()
        self.t_prev = 0.0
        self.live = None
        self.config = config

    def make_table(self, data, filepath, file_rows):
        freq = 1.0 / (data["Sec"] - self.t_prev) if self.t_prev else 0
        self.t_prev = data["Sec"]

        table = Table(title="[bold green]Certus IMU Data[/bold green]")
        table.add_column("Field", justify="left")
        table.add_column("Value", justify="right")

        table.add_row("Update rate (Hz)", f"{freq:.2f}")
        table.add_row("Timestamp", f"{data['Sec']:.5f}")
        table.add_row("", "")
        table.add_row("Latitude", f"{data['Lat']:.5f}")
        table.add_row("Longitude", f"{data['Lon']:.5f}")
        table.add_row("Altitude", f"{data['Alt']:.1f} m")
        table.add_row(
            "DGNSS heading active",
            "Yes" if data.get("DGNSS_heading_active", False) else "No",
        )
        table.add_row("", "")
        table.add_row("Roll", f"{data['Roll']:.2f}")
        table.add_row("Pitch", f"{data['Pitch']:.2f}")
        table.add_row("Heading", f"{data['Head']:.2f}")
        table.add_row("", "")
        table.add_row("Roll rate", f"{data['w_Roll']:.2f} °/s")
        table.add_row("Pitch rate", f"{data['w_Pitch']:.2f} °/s")
        table.add_row("Yaw rate", f"{data['w_Head']:.2f} °/s")
        table.add_row("", "")
        table.add_row("Current log file", f"{filepath}")
        table.add_row("Logged rows", f"{file_rows} / {self.config['log_max_rows']}")
        table.add_row(
            "Broadcast address", f"{self.config['protocol']}://{self.config['pub_address']}"
        )
        return table

    def __enter__(self):
        self.live = Live(
            self.make_table(
                {
                    "Sec": 0,
                    "Lat": 0,
                    "Lon": 0,
                    "Alt": 0,
                    "Roll": 0,
                    "Pitch": 0,
                    "Head": 0,
                    "w_Roll": 0,
                    "w_Pitch": 0,
                    "w_Head": 0,
                },
                "N/A",
                0,
            ),
            console=self.console,
            refresh_per_second=20,
            screen=True,
        )
        self.live.__enter__()
        return self

    def update(self, data, filepath="N/A", file_rows=0):
        if self.live:
            self.live.update(self.make_table(data, filepath, file_rows))

    def __exit__(self, *exc):
        if self.live:
            self.live.__exit__(*exc)


class CertusBroadcaster:
    proc: subprocess.Popen
    config: dict
    context: zmq.Context
    socket: zmq.Socket
    log_file: str
    log_dir: str
    log_idx: int
    log_row_idx: int
    t_prev: float

    def __init__(self, config_path: str) -> None:
        """Class to handle logging and broadcasting Certus IMU data."""

        self.log_idx = 0
        self.log_row_idx = 0
        self.t_prev = time.time()

        # Read config
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Open socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.set_hwm(1)
        self.socket.setsockopt(zmq.LINGER, 0)
        if self.config["protocol"] == "IPC":
            self.socket.bind(f"ipc://{self.config['pub_address']}")
        elif self.config["protocol"] == "TCP":
            self.socket.bind(f"tcp://{self.config['pub_address']}")
        else:
            raise ValueError("Unsupported protocol in config")

        # Create initial log file
        self.create_log_file(self.log_idx)

        # Start certus executable
        self.reconnect_loop(initial=True)

    def read_config(self, config_path: str) -> None:
        """Read configuration from YAML file."""

        default = {
            "log_raw": False,
            "log_dir": "~/logs_imu",
            "log_max_rows": 100000,
            "baudrate": 115200,
            "protocol": "IPC",
            "pub_address": "/tmp/imu.sock",
            "sub_address:": "/tmp/imu.sock",
            "serials": [
                "AV0LF8X0",
            ],
        }

        if any(not key in self.config for key in default):
            print("Updating config file with missing default entries...")
            with open(config_path, "w") as file:
                yaml.dump({**default, **self.config}, file)

        # Add missing entries with default
        with open(config_path, "r") as file:
            self.config = {**default, **yaml.safe_load(file)}

    def reconnect_loop(self, timeout=1, initial=False) -> None:
        """Attempt to reconnect every `timeout` seconds."""
        while True:
            if self.reconnect(initial=initial):
                print("Reconnected to Certus IMU.")
                return
            print(f"Reconnection failed. Retrying in {timeout} seconds...")
            time.sleep(timeout)

    def reconnect(self, initial=False) -> bool:
        """Reconnect to the Certus IMU."""
        try:
            if not initial:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    self.proc.wait()

            ser_port = port_serial_search(self.config["serials"])
            if ser_port is None:
                raise RuntimeError("Could not find Certus IMU. Is it connected?")

            self.proc = subprocess.Popen(
                [
                    CERTUS_EXEC,
                    ser_port,
                    str(self.config["baudrate"]),
                    str(int(self.config["log_raw"])),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            return True
        except Exception as e:
            print(f"Reconnection failed: {e}")
            return False

    def create_log_file(self, idx: int = 0) -> None:
        """Create a new log file with suffix idx."""

        time_str = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.expanduser(self.config["log_dir"])
        self.log_file = os.path.join(self.log_dir, f"{time_str}_{idx}.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write("timestamp,Lat,Lon,Alt,Roll,Pitch,Heading\n")
        print(f"Logging to {self.log_file}")

    def append_to_log(self, data: dict) -> None:
        """Append a line to the log file."""

        with open(self.log_file, "a") as f:
            f.write(
                f"{data['Sec']},{data['Lat']},{data['Lon']},{data['Alt']},{
                    data['Roll']
                },{data['Pitch']},{data['Head']}\n"
            )
        self.log_row_idx += 1
        if self.log_row_idx >= self.config["log_max_rows"]:
            self.log_idx += 1
            self.log_row_idx = 0
            self.create_log_file(self.log_idx)

    def broadcast(self, data: dict) -> None:
        """Broadcast data over ZMQ socket."""
        self.socket.send_json(data)

    def run(self) -> None:
        """Main loop to read and broadcast data."""

        if self.proc is None or self.proc.stdout is None:
            self.reconnect_loop()

        with IMUDisplay(self.config) as imu_display:
            while True:
                ready, _, _ = select.select([self.proc.stdout], [], [], 0.5)
                if ready:
                    try:
                        line = self.proc.stdout.readline()
                    except Exception as e:
                        self.reconnect_loop()
                        continue
                    try:
                        data = json.loads(line)
                        data["PC_Time"] = time.time()
                        self.broadcast(data)
                        imu_display.update(data, self.log_file, self.log_row_idx)
                        self.append_to_log(data)
                    except Exception as e:
                        pass
                else:
                    self.reconnect_loop()


if __name__ == "__main__":
    default_config_path = os.path.join(os.path.dirname(__file__), "certus_config.yaml")
    broadcaster = CertusBroadcaster(config_path=default_config_path)
    broadcaster.run()
