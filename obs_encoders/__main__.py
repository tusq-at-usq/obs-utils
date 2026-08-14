
from pathlib import Path
import sys
import os
from .run import EncoderBroadcaster

def main():
    default_config_path = os.path.join(
        Path(__file__).resolve().parent,
        "encoder_config.yaml",
    )
    broadcaster = EncoderBroadcaster(config_path=default_config_path)

    _ = sys.stdout.write(f"\33]0;{'Encoder logging and streaming'}\a")
    _ = sys.stdout.flush()

    broadcaster.run()


if __name__ == "__main__":
    main()
