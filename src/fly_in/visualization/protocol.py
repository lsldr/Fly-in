"""Protocol for renderers that observe simulation state."""

from typing import Protocol

from fly_in.simulation import Simulation


class Visualizer(Protocol):
    """Rendering protocol for a completed simulation state.

    Implementations should display the supplied state without stepping it.
    """

    def render(self, simulation: Simulation) -> None:
        """Render the current state without advancing the simulation.

        Args:
            simulation (Simulation): Simulation state to display.
        """
        ...
