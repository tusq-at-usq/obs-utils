import pyudev
from typing import Sequence


def list_usb_tty_devices() -> list[dict]:
    """List connected USB tty devices with common identifying attributes."""
    context = pyudev.Context()
    devices: list[dict] = []
    for device in context.list_devices(subsystem="tty"):
        if not device.device_node:
            continue
        parent = device.find_parent("usb", "usb_device")
        if parent is None:
            continue

        serial_short = (
            device.get("ID_SERIAL_SHORT")
            or parent.get("ID_SERIAL_SHORT")
            or ""
        )
        serial_full = device.get("ID_SERIAL") or parent.get("ID_SERIAL") or ""
        vendor_id = parent.get("ID_VENDOR_ID") or ""
        model_id = parent.get("ID_MODEL_ID") or ""

        devices.append(
            {
                "port": device.device_node.split("/")[-1],
                "serial_short": serial_short,
                "serial_full": serial_full,
                "vidpid": f"{vendor_id}:{model_id}" if vendor_id and model_id else "",
            }
        )
    return devices


def port_single_usb_tty() -> str | None:
    """Return a tty port if exactly one USB tty device is present."""
    devices = list_usb_tty_devices()
    if len(devices) == 1:
        return devices[0]["port"]
    return None

def show_current_vidpid() -> None:
    """Show VID:PID of connected USB devices."""
    context = pyudev.Context()
    for device in context.list_devices(subsystem="tty"):
        parent = device.find_parent("usb", "usb_device")
        if parent is None:
            continue
        if parent.get("ID_VENDOR_ID") and parent.get("ID_MODEL_ID"):
            print(
                f"Device: {device.device_node}, VID:PID = {parent.get('ID_VENDOR_ID')}:{parent.get('ID_MODEL_ID')}"
            )

def show_current_serials() -> None:
    """Show serial numbers of connected USB devices."""
    context = pyudev.Context()
    for device in context.list_devices(subsystem="tty"):
        parent = device.find_parent("usb", "usb_device")
        if parent is None:
            continue
        if parent.get("ID_SERIAL_SHORT"):
            print(
                f"Device: {device.device_node}, Serial Number = {parent.get('ID_SERIAL_SHORT')}"
            )


def port_by_vidpid(vidpid: str) -> str | None:
    """Return /dev/tty* port for given USB VID:PID."""
    context = pyudev.Context()
    for device in context.list_devices(subsystem="tty"):
        parent = device.find_parent("usb", "usb_device")
        if parent is None:
            continue
        if parent.get("ID_VENDOR_ID") and parent.get("ID_MODEL_ID"):
            if f"{parent.get('ID_VENDOR_ID')}:{parent.get('ID_MODEL_ID')}" == vidpid:
                return device.device_node.split("/")[-1]
    return None

def port_by_serial(serial: str) -> str | None:
    """Return /dev/tty* port for given USB serial number."""
    serial_lower = serial.strip().lower()
    for device in list_usb_tty_devices():
        if not serial_lower:
            continue
        if device["serial_short"].lower() == serial_lower:
            return device["port"]
        if device["serial_full"].lower() == serial_lower:
            return device["port"]
    return None

def port_serial_search(serials: Sequence[str]) -> str | None:
    """ Search for a port from a tuple of serial numbers."""
    for serial in serials:
        port = port_by_serial(serial)
        if port is not None:
            return port
    return None

def print_search():
    show_current_vidpid()
    show_current_serials()

if __name__ == "__main__":
    print_search()
