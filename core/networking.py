from __future__ import annotations

import re
import shutil
import subprocess


def local_ipv4_addresses() -> set[str]:
    """Return IPv4 addresses assigned to local network interfaces."""
    addresses = {"127.0.0.1"}
    if shutil.which("ifconfig"):
        try:
            result = subprocess.run(["ifconfig"], capture_output=True, text=True, check=False)
        except OSError:
            return addresses
        addresses.update(re.findall(r"^\s*inet\s+(\d+\.\d+\.\d+\.\d+)", result.stdout, flags=re.MULTILINE))
    elif shutil.which("ip"):
        try:
            result = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, check=False)
        except OSError:
            return addresses
        addresses.update(re.findall(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/", result.stdout))
    elif shutil.which("ipconfig"):
        try:
            result = subprocess.run(["ipconfig"], capture_output=True, text=True, check=False)
        except OSError:
            return addresses
        addresses.update(re.findall(r"^\s*IPv4 [^:\r\n]*:\s*(\d+\.\d+\.\d+\.\d+)", result.stdout, flags=re.MULTILINE))
    return addresses
