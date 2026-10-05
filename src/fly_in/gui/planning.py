"""Prepare an isolated simulation outside the Qt event thread."""

from threading import Event
from time import perf_counter
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from fly_in.solver_factory import SolverFactory
from fly_in.models import MapConfig
from fly_in.provenance import PlanningProvenance
from fly_in.simulation import Simulation
from fly_in.solvers.protocol import PreparableSolver, Solver


class PlanningWorker(QThread):
    """Transfer a prepared simulation only after all planning mutations end."""

    prepared = Signal(int, object, float)
    failed = Signal(int, str)

    def __init__(
        self, generation: int, config: MapConfig, solver_name: str,
        solver_parameters: dict[str, float | int] | None = None,
        map_name: str = 'in_memory', records_dir: Path | None = None,
    ) -> None:
        """Capture an isolated input and cancellation token."""
        super().__init__()
        self.generation = generation
        self.config = config.model_copy(deep=True)
        self.solver_name = solver_name
        self.solver_parameters = dict(solver_parameters or {})
        self.map_name = map_name
        self.records_dir = records_dir
        self.cancelled = Event()
        self.planning_provenance = PlanningProvenance()
        self.planning_duration_seconds = 0.0

    def cancel(self) -> None:
        """Ask the planner to stop at its next cancellation checkpoint."""
        self.cancelled.set()

    def run(self) -> None:
        """Build and prepare isolated state, then emit a queued result."""
        started = perf_counter()
        solver: Solver | None = None
        try:
            solver = SolverFactory(
                self.map_name, self.records_dir,
            ).create(self.solver_name, self.solver_parameters)
            simulation = Simulation.from_config(self.config, solver)
            if isinstance(solver, PreparableSolver):
                solver.prepare(
                    simulation.network, simulation.drones,
                    self.cancelled.is_set,
                )
            self.planning_duration_seconds = perf_counter() - started
            self.planning_provenance = PlanningProvenance.capture(solver)
            if not self.cancelled.is_set():
                self.prepared.emit(
                    self.generation, simulation, self.planning_duration_seconds
                )
        except (RuntimeError, ValueError, OSError) as error:
            self.planning_duration_seconds = perf_counter() - started
            self.planning_provenance = PlanningProvenance.capture(
                solver, error,
            )
            if not self.cancelled.is_set():
                self.failed.emit(self.generation, str(error))
