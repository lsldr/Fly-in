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
    """Background Qt thread for preparing isolated simulation state.

    Signals:
        prepared (Signal[int, object, float]): Emits generation, result, and
            elapsed seconds after successful preparation.
        failed (Signal[int, str]): Emits generation and a planning error.

    Attributes:
        generation (int): Identifier used to reject stale worker results.
        config (MapConfig): Deep copy of the requested map configuration.
        solver_name (str): Registered solver name selected for planning.
        solver_parameters (dict[str, float | int]): Solver factory options.
        map_name (str): Name used by solver-specific records.
        records_dir (Path | None): Optional directory for solver records.
        cancelled (Event): Set when the worker should suppress its result.
        planning_provenance (PlanningProvenance): Captured status and proof.
        planning_duration_seconds (float): Time spent preparing simulation.
    """

    prepared = Signal(int, object, float)
    failed = Signal(int, str)

    def __init__(
        self, generation: int, config: MapConfig, solver_name: str,
        solver_parameters: dict[str, float | int] | None = None,
        map_name: str = 'in_memory', records_dir: Path | None = None,
    ) -> None:
        """Capture an isolated input and cancellation token.

        Args:
            generation (int): Caller-assigned identifier for this planning
                request.
            config (MapConfig): Map configuration to deep-copy before planning.
            solver_name (str): Solver registered with the factory.
            solver_parameters (dict[str, float | int] | None): Optional numeric
                solver options.
            map_name (str): Name used for solver records; defaults to
                ``in_memory``.
            records_dir (Path | None): Optional directory for persisted solver
                records.
        """
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
        """Set the cancellation event checked by cooperative planners."""
        self.cancelled.set()

    def run(self) -> None:
        """Build isolated state and emit its result through Qt signals.

        The ``prepared`` or ``failed`` signal is emitted only if cancellation
        was not requested. Factory, preparation, and construction failures of
        type RuntimeError, ValueError, or OSError are reported through
        ``failed``; the worker thread itself does not propagate them.
        """
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
