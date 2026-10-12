"""Structural interfaces and schedule aliases for drone planners."""

from collections.abc import Callable
from typing import Protocol, TypeAlias, runtime_checkable

from fly_in.realm.entities import Drone
from fly_in.realm.network import DroneNetwork

Schedule: TypeAlias = list[list[str]]
"""Actions grouped by turn; each action is a drone departure command."""


class Solver(Protocol):
    """Interface for a solver that plans drone movements."""

    def plan(self, network: DroneNetwork, drones: list[Drone]) -> Schedule:
        """Return the next turn's departure actions.

        Args:
            network (DroneNetwork): Topology on which drones are traveling.
            drones (list[Drone]): Current fleet state.

        Returns:
            Schedule: Turn rows, with the first row ready for execution.
        """
        ...


@runtime_checkable
class PreparableSolver(Solver, Protocol):
    """A solver that can plan off the playback thread before the first turn."""

    def prepare(
        self, network: DroneNetwork, drones: list[Drone],
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Prepare a schedule for a fresh fleet without advancing a turn.

        Args:
            network (DroneNetwork): Topology for the plan.
            drones (list[Drone]): Undelivered fleet at the start hub.
            cancelled (Callable[[], bool] | None): Optional cooperative
                cancellation check.
        """
        ...
