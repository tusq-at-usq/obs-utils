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
import serial

from obs_utils.discovery import port_serial_search


class EncoderDisplay:
    def __init__(self, config: dict) -> None:
        self.console = Console()
        self.t_prev = 0.0
        self.live = None
        self.config = config

    def make_table(self, data, filepath, file_rows):
        freq = 1.0 / (data["Sec"] - self.t_prev) if self.t_prev else 0
        self.t_prev = data["Sec"]

        table = Table(title="[bold green]Encoder Data[/bold green]")
        table.add_column("Field", justify="left")
        table.add_column("Value", justify="right")

        table.add_row("Update rate (Hz)", f"{freq:.2f}")
        table.add_row("Timestamp", f"{data['Sec']:.5f}")
        table.add_row("", "")
        table.add_row("Azimimuth", f"{data['Az']:.3f}")
        table.add_row("Elevation", f"{data['El']:.3f}")
        table.add_row("","")
        table.add_row("Azimimuth (raw)", f"{data['Az_raw']:.1f}")
        table.add_row("Elevation (raw)", f"{data['El_raw']:.1f}")
        table.add_row("", "")
        table.add_row("Log file", f"{filepath}")
        table.add_row("Log rows", f"{file_rows}")
        return table

    def __enter__(self):
        self.live = Live(
            self.make_table(
                {
                    "Sec": 0,
                    "Az": 0,
                    "El": 0,
                    "Az_raw": 0,
                    "El_raw": 0,
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


class EncoderBroadcaster:
    ser: serial.Serial
    config: dict
    context: zmq.Context
    socket: zmq.Socket
    log_file: str
    log_dir: str
    log_idx: int
    log_row_idx: int
    t_prev: float

    def __init__(self, config_path: str) -> None:
        """Class to handle logging and broadcasting encoder data."""

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
            self.socket.bind(f"tcp://*:{self.config['pub_address']}")
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
            "log_dir": "~/logs_encoder",
            "log_max_rows": 100000,
            "serials": [
                "A703YD9I",
            ],
            "baudrate": 115200,
            "protocol": "IPC",
            "address": "/tmp/encoder.sock",
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
                print("Reconnected to encoders.")
                return
            print(f"Reconnection failed. Retrying in {timeout} seconds...")
            time.sleep(timeout)

    def reconnect(self, initial=False) -> bool:
        """Reconnect to encoders."""
        try:
            if not initial:
                if self.ser.is_open:
                    self.ser.close()

            ser_port = port_serial_search(self.config["serials"])
            if ser_port is None:
                raise RuntimeError("Could not find encoder interface. Is it connected?")

            self.ser = serial.Serial(os.path.join("/dev", ser_port), self.config["baudrate"], timeout=1)
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
            f.write("timestamp,Az,El,Az_raw,El_raw\n")
        print(f"Logging to {self.log_file}")

    def dict_from_line(self, t_unix: float, line: str) -> dict:
        """Convert a line of text to a JSON dict."""
        parts = line.split(",")
        print(line)
        print(parts)
        data = {
            "Sec": t_unix,
            "Az": float(parts[2]),
            "El": float(parts[3]),
            "Az_raw": float(parts[0]),
            "El_raw": float(parts[1]),
        }
        return data

    def append_to_log(self, data: dict) -> None:
        """Append a line to the log file."""

        with open(self.log_file, "a") as f:
            f.write(
                f"{data['Sec']},{data['Az']},{data['El']},{data['Az_raw']},{data['El_raw']}\n"
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

        if self.ser is None or self.ser.is_open is False:
            self.reconnect_loop()

        killswitch = 0

        with EncoderDisplay(self.config) as encoder_display:
            while not killswitch:
                try:
                    text = self.ser.readline().strip()
                except KeyboardInterrupt:
                    killswitch=1
                    continue
                except:
                    self.reconnect_loop()
                    continue
                t = time.time()
                try:
                    text = text.decode("utf8")
                    data = self.dict_from_line(t, text)
                except KeyboardInterrupt:
                    killswitch = 1
                    continue
                except:
                    continue
                self.broadcast(data)
                encoder_display.update(data, self.log_file, self.log_row_idx)
                self.append_to_log(data)
        print("Exited encoder script gracefully")


if __name__ == "__main__":
    default_config_path = os.path.join(os.path.dirname(__file__), "encoder_config.yaml")
    broadcaster = EncoderBroadcaster(config_path=default_config_path)
    broadcaster.run()
