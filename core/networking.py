from __future__ import annotations

import ipaddress
import socket


def local_ipv4_addresses() -> set[str]:
    """Best-effort set of IPv4 addresses assigned to this computer."""
    addresses = {"127.0.0.1"}
    names = {socket.gethostname(), socket.getfqdn(), "localhost"}
    for name in names:
        try:
            for result in socket.getaddrinfo(name, None, socket.AF_INET):
                addresses.add(str(ipaddress.IPv4Address(result[4][0])))
        except (OSError, ValueError):
            continue
    try:
        import psutil

        for interface in psutil.net_if_addrs().values():
            for address in interface:
                if address.family == socket.AF_INET:
                    addresses.add(str(ipaddress.IPv4Address(address.address)))
    except (ImportError, OSError, ValueError):
        pass
    return addresses


def validate_ipv4(value: str, *, label: str = "IP address", allow_unspecified: bool = False) -> str:
    try:
        address = ipaddress.IPv4Address(value.strip())
    except (ipaddress.AddressValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a valid IPv4 address.") from exc
    if address.is_unspecified and not allow_unspecified:
        raise ValueError(f"{label} cannot be 0.0.0.0.")
    return str(address)
