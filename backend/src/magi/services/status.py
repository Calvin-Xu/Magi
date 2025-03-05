import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, NamedTuple


class ServiceState(Enum):
    """Possible states for a service."""

    UNKNOWN = auto()
    OK = auto()
    ERROR = auto()

    def to_emoji(self) -> str:
        """Convert state to an emoji representation."""
        return {
            ServiceState.OK: "✅",
            ServiceState.ERROR: "❌",
            ServiceState.UNKNOWN: "⚪",
        }[self]


class ServiceInfo(NamedTuple):
    """Information about a service's status."""

    state: ServiceState
    message: str


@dataclass
class ServiceStatus:
    """Thread-safe service status tracker."""

    _lock: threading.Lock
    _status: Dict[str, ServiceInfo]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = {
            "memgraph": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
            "postgres_(pgvector)": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
            "spark_local": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
            "memgraph_lab": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
        }

    def update(self, service: str, state: ServiceState, message: str) -> None:
        """Update the status of a service."""
        with self._lock:
            self._status[service] = ServiceInfo(state, message)

    def get_all(self) -> Dict[str, ServiceInfo]:
        """Get a copy of all service statuses."""
        with self._lock:
            return dict(self._status)


# Global service status instance
service_status = ServiceStatus()
