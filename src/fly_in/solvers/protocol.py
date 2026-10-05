from collections.abc import Callable
from typing import Protocol, TypeAlias, runtime_checkable

from fly_in.realm.entities import Drone
from fly_in.realm.network import DroneNetwork

"""A turn schedule whose first row contains the current turn's actions."""

Schedule: TypeAlias = list[list[str]]


class Solver(Protocol):
    """Interface for a solver that plans drone movements."""

    def plan(self, network: DroneNetwork, drones: list[Drone]) -> Schedule:
        """Return a schedule whose first row contains current departures."""
        ...


@runtime_checkable
class PreparableSolver(Solver, Protocol):
    """A solver that can plan off the playback thread before the first turn."""

    def prepare(
        self, network: DroneNetwork, drones: list[Drone],
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Cache a complete schedule for this fresh fleet."""
        ...
