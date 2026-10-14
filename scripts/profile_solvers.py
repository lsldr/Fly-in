"""Measure solver planning time, peak traced memory, and search counters."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
import tracemalloc

from fly_in.solver_factory import SOLVER_CHOICES, SolverFactory
from fly_in.output import AtomicFileWriter
from fly_in.parser import Parser
from fly_in.provenance import PlanningProvenance
from fly_in.simulation import Simulation
from fly_in.solvers.base import ScheduledSolver
from fly_in.solvers.protocol import PreparableSolver


class SolverProfiler:
    """Planning measurements for selected strategies on one map.

    Attributes:
        map_path (Path): Source map to parse once for the measurements.
        strategies (tuple[str, ...]): Solver names in measurement order.
        time_budget (float): Planning budget in seconds passed to each solver.
        records_root (Path): Directory for solver-generated records.
    """

    def __init__(
        self,
        map_path: Path,
        strategies: list[str],
        time_budget: float,
        records_root: Path,
    ) -> None:
        """Capture the profiling inputs without parsing or executing the map.

        Args:
            map_path (Path): Source map file.
            strategies (list[str]): Solver names to measure, copied into a
                tuple.
            time_budget (float): Planning budget in seconds for each strategy.
            records_root (Path): Destination for solver-generated records.
        """
        self.map_path = map_path
        self.strategies = tuple(strategies)
        self.time_budget = time_budget
        self.records_root = records_root

    def measure(self) -> dict[str, object]:
        """Measure preparation time and memory, then execute each schedule.

        Create a fresh simulation per strategy. Include execution failures in
        result rows, and preserve planning provenance separately. Start and
        stop
        process-wide tracemalloc tracing around each preparation; execution is
        excluded from planning time and peak traced memory.

        Returns:
            dict[str, object]: JSON-compatible report with schema version, UTC
            timestamp, source path, environment, and ordered result rows. Rows
            include planning milliseconds, peak traced bytes, search counters,
            and either execution metrics or a failure message.

        Raises:
            ConfigParseError: The source map cannot be read or parsed.
            ValueError: Solver creation or simulation setup rejects its inputs.
        """
        config = Parser().parse(self.map_path)
        results = []
        for name in self.strategies:
            solver = SolverFactory(
                self.map_path.stem,
                self.records_root,
            ).create(name, {"time_budget": self.time_budget})
            simulation = Simulation.from_config(config, solver)
            tracemalloc.start()
            started = perf_counter()
            planning: PlanningProvenance | None = None
            row: dict[str, object] = {"solver": name}
            try:
                try:
                    if isinstance(solver, PreparableSolver):
                        solver.prepare(simulation.network, simulation.drones)
                finally:
                    row["planning_ms"] = round(
                        (perf_counter() - started) * 1000,
                        3,
                    )
                    _, peak = tracemalloc.get_traced_memory()
                    row["tracemalloc_peak_bytes"] = peak
                    tracemalloc.stop()
                planning = PlanningProvenance.capture(solver)
                result = simulation.run()
                row.update(
                    {
                        "status": "completed",
                        "turns": result.turns,
                        "physical_path_cost": result.metrics.total_path_cost,
                    }
                )
            except (RuntimeError, ValueError, OSError) as error:
                row.update({"status": "failed", "error": str(error)})
                if planning is None:
                    planning = PlanningProvenance.capture(solver, error)
            solution = (
                solver.solution
                if isinstance(
                    solver,
                    ScheduledSolver,
                )
                else None
            )
            row.update(
                {
                    "planning": planning.record(),
                    "expanded": solution.stats.expanded if solution else None,
                    "high_level_nodes": (
                        solution.stats.nodes if solution else None
                    ),
                    "iterations": (
                        solution.stats.iterations if solution else None
                    ),
                }
            )
            results.append(row)
        return {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "map_path": str(self.map_path.resolve()),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "results": results,
        }


class ProfilingApplication:
    """Command-line options and report persistence for solver profiling."""

    def run(self) -> None:
        """Profile the command-line map and save a timestamped JSON report.

        Print the report and its path to stdout. Create the record directory as
        needed, and use fly_in.output.AtomicFileWriter to replace the report.

        Raises:
            SystemExit: Help was requested (code 0), or arguments are invalid
                (code 2), including a negative time budget.
            ConfigParseError: The source map cannot be read or parsed.
            ValueError: Solver creation or simulation setup rejects its inputs.
            OSError: The report cannot be persisted.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("map", type=Path)
        parser.add_argument(
            "--solver",
            action="append",
            choices=SOLVER_CHOICES,
        )
        parser.add_argument("--time-budget", type=float, default=1.0)
        parser.add_argument(
            "--records-root", type=Path, default=Path("records")
        )
        arguments = parser.parse_args()
        if arguments.time_budget < 0:
            parser.error("--time-budget must be nonnegative")
        report = SolverProfiler(
            arguments.map,
            arguments.solver or ["priority", "lns", "pbs", "icbs", "adaptive"],
            arguments.time_budget,
            arguments.records_root,
        ).measure()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = arguments.records_root / f"profile_{stamp}.json"
        AtomicFileWriter(destination).write(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))
        print(f"Report: {destination}")


def main() -> None:
    """Run the profiling application using sys.argv.

    Raises:
        SystemExit: Help was requested or command-line arguments are invalid.
        ConfigParseError: The source map cannot be read or parsed.
        ValueError: Solver creation or simulation setup rejects its inputs.
        OSError: The report cannot be persisted.
    """
    ProfilingApplication().run()


if __name__ == "__main__":
    main()
