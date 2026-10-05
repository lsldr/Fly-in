from typing import Protocol

from fly_in.simulation import Simulation


class Visualizer(Protocol):
    """Interface for rendering a completed simulation state."""

    def render(self, simulation: Simulation) -> None:
        """Render the current state without advancing the simulation."""
        ...
