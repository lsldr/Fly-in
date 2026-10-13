"""Reproducible map benchmarks with machine-readable provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from collections.abc import Callable, Mapping

from rich.console import Console
from rich.table import Table

from fly_in.output import AtomicFileWriter
from fly_in.solver_factory import (
    DEFAULT_SOLVER, SOLVER_CHOICES, SolverFactory,
)
from fly_in.parser import Parser
from fly_in.provenance import PlanningProvenance
from fly_in.simulation import Simulation
from fly_in.solvers.protocol import PreparableSolver, Solver

TARGETS = {
    "easy/01_linear_path.txt": 6,
    "easy/02_simple_fork.txt": 8,
    "easy/03_basic_capacity.txt": 6,
    "medium/01_dead_end_trap.txt": 12,
    "medium/02_circular_loop.txt": 15,
    "medium/03_priority_puzzle.txt": 12,
    "hard/01_maze_nightmare.txt": 30,
    "hard/02_capacity_hell.txt": 35,
    "hard/03_ultimate_challenge.txt": 45,
    "challenger/01_the_impossible_dream.txt": 45,
}


class BenchmarkRunner:
    """Runner for a map catalog with isolated solvers and reports.

    Attributes:
        map_root (Path): Resolved root recursively searched for map files.
        solver_name (str): Registered strategy used for each map.
        records_root (Path): Destination directory for the benchmark report.
        solver_options (dict[str, float | int]): Copied strategy parameters.
    """

    def __init__(
        self, map_root: str | Path = "maps",
        solver_name: str = DEFAULT_SOLVER,
        records_root: str | Path = "records",
        solver_options: Mapping[str, float | int] | None = None,
    ) -> None:
        """Bind the catalog, strategy, report destination, and options.

        Args:
            map_root (str | Path): Directory recursively searched for maps.
            solver_name (str): Registered strategy used for each map.
            records_root (str | Path): Directory receiving the JSON report.
            solver_options (Mapping[str, float | int] | None): Optional
                solver settings copied before use.
        """
        self.map_root = Path(map_root).resolve()
        self.solver_name = solver_name
        self.records_root = Path(records_root)
        self.solver_options = dict(solver_options or {})

    def run(
        self, on_progress: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, object], Path]:
        """Run each map once and retain individual failures in the report.

        Args:
            on_progress (Callable[[str], None] | None): Optional listener for
                map-start and map-finish messages, including failed maps.

        Returns:
            tuple[dict[str, object], Path]: Report data and its saved path.

        Raises:
            ValueError: If the strategy is unknown or the catalog contains no
                text maps.
            OSError: If the benchmark report cannot be written. Map read
                failures are recorded in individual result rows.
        """
        if self.solver_name not in SOLVER_CHOICES:
            raise ValueError(f"unknown solver '{self.solver_name}'")
        options = dict(self.solver_options)
        root = self.map_root
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix == ".txt"
        )
        if not paths:
            raise ValueError(f"no .txt files found under {root}")
        results: list[dict[str, object]] = []
        for index, path in enumerate(paths, 1):
            relative = path.relative_to(root).as_posix()
            if on_progress is not None:
                on_progress(f'[{index}/{len(paths)}] Starting {relative}')
            target = TARGETS.get(relative)
            if target is None and root.name in {
                'easy', 'medium', 'hard', 'challenger',
            }:
                target = TARGETS.get(f'{root.name}/{relative}')
            row: dict[str, object] = {"map": relative, "target_turns": target}
            started = perf_counter()
            solver: Solver | None = None
            planning: PlanningProvenance | None = None
            try:
                row["map_sha256"] = hashlib.sha256(
                    path.read_bytes(),
                ).hexdigest()
                config = Parser().parse(path)
                row['nb_drones'] = config.nb_drones
                solver = SolverFactory(
                    relative, self.records_root,
                ).create(self.solver_name, options)
                simulation = Simulation.from_config(config, solver)
                planning_started = perf_counter()
                if isinstance(solver, PreparableSolver):
                    solver.prepare(simulation.network, simulation.drones)
                row["planning_duration_seconds"] = (
                    perf_counter() - planning_started
                )
                planning = PlanningProvenance.capture(solver)
                result = simulation.run()
                row.update(
                    {
                        "status": "completed",
                        "turns": result.turns,
                        "total_path_cost": result.metrics.total_path_cost,
                        "winning_rung": getattr(solver, 'winning_rung', None),
                        "meets_target": (
                            (result.turns < target
                             if (relative.startswith("challenger/")
                                 or root.name == 'challenger')
                             else result.turns <= target)
                            if target is not None else None
                        ),
                    }
                )
            except (OSError, ValueError, RuntimeError) as error:
                planning = planning or PlanningProvenance.capture(
                    solver, error,
                )
                row.update({
                    "status": "failed", "error": str(error),
                })
            row.update({
                'planning': planning.record(),
                'search_status': planning.status,
                'optimal_makespan': planning.optimal_makespan,
            })
            duration = perf_counter() - started
            row["duration_seconds"] = duration
            results.append(row)
            if on_progress is not None:
                on_progress(
                    f'[{index}/{len(paths)}] {relative}: '
                    f'{row["status"]} ({duration:.3f}s)'
                )
        timestamp = datetime.now(timezone.utc)
        report: dict[str, object] = {
            "schema_version": 1,
            "timestamp": timestamp.isoformat(),
            "solver": self.solver_name,
            "solver_parameters": options,
            "map_root": str(root),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "results": results,
        }
        name = timestamp.strftime("benchmark_%Y%m%dT%H%M%S%fZ.json")
        report_path = Path(self.records_root) / name
        AtomicFileWriter(report_path).write(
            json.dumps(report, indent=2) + "\n"
        )
        return report, report_path


class BenchmarkApplication:
    """Benchmark command parser and terminal result presenter."""

    def run(self, argv: list[str]) -> int:
        """Execute benchmark arguments and present the saved report.

        Args:
            argv (list[str]): Benchmark arguments excluding the command name.

        Returns:
            int: Process status, zero when all benchmark targets pass.
        """
        parser = argparse.ArgumentParser(prog="fly-in benchmark")
        parser.add_argument("--map-root", type=Path, default=Path("maps"))
        parser.add_argument(
            "--records-root", type=Path, default=Path("records"),
        )
        parser.add_argument(
            "--solver", choices=SOLVER_CHOICES, default=DEFAULT_SOLVER
        )
        parser.add_argument('--time-budget', type=float, default=None)
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--max-iter', type=int, default=None)
        parser.add_argument('--node-budget', type=int, default=None)
        args = parser.parse_args(argv)
        console = Console()

        def progress(message: str) -> None:
            """Print each map's progress before waiting for the next result.

            Args:
                message (str): Plain-text benchmark progress message.
            """
            console.print(message, markup=False)

        try:
            report, path = BenchmarkRunner(
                args.map_root, args.solver, args.records_root,
                {key: value for key, value in (
                    ('time_budget', args.time_budget), ('seed', args.seed),
                    ('max_iter', args.max_iter),
                    ('node_budget', args.node_budget),
                ) if value is not None},
            ).run(on_progress=progress)
        except (OSError, ValueError) as error:
            console.print(f"Benchmark failed: {error}", markup=False)
            return 1
        table = Table("Map", "Turns", "Path cost", "Target", "Result",
                      "Plan proof")
        results = report["results"]
        assert isinstance(results, list)
        failed = False
        for row in results:
            status = row["status"]
            meets = row.get("meets_target")
            outcome = "PASS" if meets else "MISS" if meets is False else status
            failed |= status == "failed" or meets is False
            table.add_row(
                str(row["map"]),
                str(row.get("turns", "—")),
                str(row.get("total_path_cost", "—")),
                str(row.get("target_turns") or "—"),
                outcome,
                'proved' if row['optimal_makespan'] else 'unproved',
            )
            if status == "failed":
                console.print(str(row.get("error")), markup=False)
        console.print(table)
        console.print(f"Report: {path}", markup=False)
        return 1 if failed else 0
