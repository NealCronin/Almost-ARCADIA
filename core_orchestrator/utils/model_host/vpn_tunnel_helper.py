"""
vpn_tunnel_helper.py

Network helpers for VPN-tunnel diagnostics.

Responsibilities
----------------
* Check host reachability via raw TCP socket connect.
* Resolve the local IP address bound to a given network interface.
* Enumerate interfaces that are UP and have a non-loopback address.
* Verify end-to-end tunnel connectivity.
"""

from __future__ import annotations

import fcntl
import logging
import socket
import struct
from typing import Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CONNECT_TIMEOUT = 3.0  # seconds


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------
def is_host_reachable(host: str, port: int, timeout: float = CONNECT_TIMEOUT) -> bool:
    """
    Attempt a raw TCP connect to *host*:*port*.

    Parameters
    ----------
    host : str
    port : int
    timeout : float
        Seconds to wait for the TCP handshake.

    Returns
    -------
    bool
        ``True`` if the connect succeeded.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except (socket.timeout, OSError):
        return False
    finally:
        sock.close()


def get_local_ip(iface: str = "eth0") -> Optional[str]:
    """
    Return the IPv4 address bound to *iface* via the ``SIOCGIFADDR`` ioctl.

    Parameters
    ----------
    iface : str
        Network interface name (e.g. ``"en0"``, ``"eth0"``).

    Returns
    -------
    str or None
        The IPv4 address, or ``None`` on failure.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # SIOCGIFADDR = 0x8020691f on Linux / Darwin
        info = fcntl.ioctl(sock.fileno(), 0x8020691f, struct.pack("256s", iface.encode("utf-8")[:15]))
        ip = socket.inet_ntoa(info[20:24])
        return ip
    except Exception as exc:
        logger.debug("Cannot resolve IP for interface '%s': %s", iface, exc)
        return None
    finally:
        sock.close()


def check_vpn_interfaces() -> dict[str, Optional[str]]:
    """
    Enumerate common VPN/tunnel interfaces and return their IPs.

    Returns
    -------
    dict
        Mapping of interface name (e.g. ``"tun0"``, ``"utun0"``) to
        IPv4 address (or ``None`` if unreachable).
    """
    common_vpn_ifaces = ["tun0", "tun1", "tun2", "utun0", "utun1", "utun2", "ppp0", "ppp1"]
    results: dict[str, Optional[str]] = {}
    for iface in common_vpn_ifaces:
        ip = get_local_ip(iface)
        if ip is not None:
            results[iface] = ip
            logger.info("VPN tunnel interface '%s' -> %s", iface, ip)
    return results


def verify_tunnel(
    host: str,
    port: int,
    require_vpn: bool = False,
    timeout: float = CONNECT_TIMEOUT,
) -> bool:
    """
    Verify that a tunnel endpoint is reachable.

    Parameters
    ----------
    host : str
    port : int
    require_vpn : bool
        If ``True``, at least one VPN interface must be present.
    timeout : float

    Returns
    -------
    bool
        ``True`` if the host is reachable and (optionally) a VPN
        interface is active.
    """
    reachable = is_host_reachable(host, port, timeout=timeout)

    if require_vpn:
        vpn_ifaces = check_vpn_interfaces()
        if not vpn_ifaces:
            logger.warning("No VPN tunnel interfaces detected")
            return False
        logger.info("VPN interfaces found: %s", list(vpn_ifaces.keys()))

    return reachable
