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

import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CONNECT_TIMEOUT = 3.0  # seconds


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------
def is_host_reachable(
    host: str,
    port: int,
    timeout: float = CONNECT_TIMEOUT,
) -> bool:
    """
    Check if a remote host:port is reachable via TCP.

    Parameters
    ----------
    host : str
        Target hostname or IP address.
    port : int
        Target port number.
    timeout : float
        Connection timeout in seconds.

    Returns
    -------
    bool
        True if the connection succeeds, False otherwise.
    """
    if not host or port <= 0:
        return False

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (socket.gaierror, socket.error) as exc:
        logger.debug("Host reachability check failed for %s:%d: %s", host, port, exc)
        return False


def get_local_ip(iface: Optional[str] = None) -> Optional[str]:
    """
    Get the local IP address for a network interface.

    Parameters
    ----------
    iface : str | None
        Interface name (e.g., "eth0", "en0"). If None, returns
        the primary outbound interface IP.

    Returns
    -------
    str | None
        Local IP address or None if not found.
    """
    try:
        if iface is None:
            # Get IP of default route interface
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]

        # Try to get specific interface IP
        import psutil

        addrs = psutil.net_if_addrs().get(iface, [])
        for addr in addrs:
            if addr.family == socket.AF_INET:
                return addr.address

    except ImportError:
        logger.warning("psutil not available, using fallback")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except socket.error:
            return None
    except Exception as exc:
        logger.debug("Failed to get local IP for %s: %s", iface, exc)

    return None


def check_vpn_interfaces() -> dict[str, Optional[str]]:
    """
    Enumerate network interfaces that might be VPN tunnels.

    Returns
    -------
    dict[str, str | None]
        Mapping of interface names to their IP addresses.
    """
    result: dict[str, Optional[str]] = {}

    try:
        import psutil

        interfaces = psutil.net_if_addrs()
        for name, addrs in interfaces.items():
            # Check for VPN-related interface names
            vpn_indicators = ["tun", "tap", "vpn", "ppp", "wireguard", "openvpn"]
            name_lower = name.lower()

            is_vpn = any(ind in name_lower for ind in vpn_indicators)

            if is_vpn:
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        result[name] = addr.address
                        break
                else:
                    result[name] = None

    except ImportError:
        logger.warning("psutil not available, cannot enumerate interfaces")
    except Exception as exc:
        logger.debug("Failed to enumerate interfaces: %s", exc)

    return result


def verify_tunnel(
    remote_host: str,
    remote_port: int,
    local_iface: Optional[str] = None,
) -> bool:
    """
    Verify end-to-end tunnel connectivity to a remote host.

    This checks:
    1. Local interface is up and has an IP
    2. Remote host:port is reachable

    Parameters
    ----------
    remote_host : str
        Target remote host.
    remote_port : int
        Target remote port.
    local_iface : str | None
        Optional local interface to use.

    Returns
    -------
    bool
        True if tunnel connectivity is verified.
    """
    # Check local interface
    if local_iface:
        local_ip = get_local_ip(local_iface)
        if not local_ip:
            logger.warning("Local interface %s not available", local_iface)
            return False
        logger.info("Local IP on %s: %s", local_iface, local_ip)

    # Check remote reachability
    reachable = is_host_reachable(remote_host, remote_port)
    if reachable:
        logger.info("Tunnel verified: %s:%d is reachable", remote_host, remote_port)
    else:
        logger.warning("Tunnel check failed: %s:%d is not reachable", remote_host, remote_port)

    return reachable


def get_network_info() -> dict:
    """
    Gather comprehensive network information.

    Returns
    -------
    dict
        Dictionary containing network configuration details.
    """
    info: dict = {
        "local_ip": get_local_ip(),
        "interfaces": {},
        "vpn_interfaces": {},
    }

    try:
        import psutil

        # Get all interfaces
        interfaces = psutil.net_if_addrs()
        for name, addrs in interfaces.items():
            ipv4_addrs = [
                addr.address
                for addr in addrs
                if addr.family == socket.AF_INET and addr.address != "127.0.0.1"
            ]
            if ipv4_addrs:
                info["interfaces"][name] = ipv4_addrs

        # Get VPN interfaces
        vpn_check = check_vpn_interfaces()
        info["vpn_interfaces"] = vpn_check

    except ImportError:
        logger.warning("psutil not available for detailed network info")

    return info


def ping_host(host: str, count: int = 3, timeout: float = 2.0) -> dict:
    """
    Ping a host and return statistics.

    Parameters
    ----------
    host : str
        Target host to ping.
    count : int
        Number of ping attempts.
    timeout : float
        Timeout per ping in seconds.

    Returns
    -------
    dict
        Ping statistics including success rate and latency.
    """
    import time

    successes = 0
    latencies = []

    for _ in range(count):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            start = time.time()
            result = sock.connect_ex((host, 80))  # Use port 80 as proxy for ICMP
            elapsed = (time.time() - start) * 1000  # ms
            sock.close()

            if result == 0:
                successes += 1
                latencies.append(elapsed)
        except socket.error:
            pass

    return {
        "host": host,
        "sent": count,
        "received": successes,
        "packet_loss": ((count - successes) / count) * 100 if count > 0 else 100,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "min_latency_ms": min(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
    }
