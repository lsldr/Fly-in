"""Named solver construction shared by terminal, desktop, and benchmarks."""

from collections.abc import Mapping
from pathlib import Path

from fly_in.solvers import (
    AdaptiveSolver, GreedySolver, ICBSSolver, LNSImprover, PBSSolver,
    PriorityBasedSolver, Solver,
)

DEFAULT_SOLVER = 'optimal'
SOLVER_CHOICES = (
    'optimal', 'priority', 'greedy', 'lns', 'pbs', 'icbs', 'adaptive',
)


class SolverFactory:
    """Factory for fresh solvers with shared source and log context.

    Attributes:
        map_name (str): Map label used by adaptive solver diagnostics.
        records_dir (Path): Directory used for solver run logs.
    """

    def __init__(
        self, map_name: str = 'in_memory', records_dir: Path | None = None,
    ) -> None:
        """Capture the map identity and directory used by adaptive logs.

        Args:
            map_name (str): Map label associated with adaptive solver logs.
            records_dir (Path | None): Log directory, defaulting to the
                current directory's ``records`` folder.
        """
        self.map_name = map_name
        self.records_dir = records_dir or Path.cwd() / 'records'

    def create(
        self, name: str = DEFAULT_SOLVER,
        options: Mapping[str, float | int] | None = None,
    ) -> Solver:
        """Build a fresh named strategy with configured search limits.

        Args:
            name (str): Registered solver key.
            options (Mapping[str, float | int] | None): Optional time, node,
                iteration, and random-seed settings. Recognized keys are
                ``time_budget`` (seconds), ``node_budget``, ``max_iter``, and
                ``seed``; other keys are ignored. Omitted optimal-search
                budgets are unlimited. Other strategies default to 1 second,
                10000 nodes, 200 iterations, and seed 0 where applicable.

        Returns:
            Solver: Newly constructed solver instance.

        Raises:
            ValueError: If ``name`` is not registered or a selected solver
                rejects one of its configured limits.
        """
        opts = options or {}
        if name == 'optimal':
            from fly_in.solvers.optimal import OptimalSolver

            return OptimalSolver(
                time_budget=(float(opts['time_budget'])
                             if 'time_budget' in opts else None),
                node_budget=(int(opts['node_budget'])
                             if 'node_budget' in opts else None),
            )
        time_budget = float(opts.get('time_budget', 1.0))
        seed = int(opts.get('seed', 0))
        max_iter = int(opts.get('max_iter', 200))
        node_budget = int(opts.get('node_budget', 10000))
        if name == "greedy":
            return GreedySolver()
        if name == "priority":
            return PriorityBasedSolver()
        if name == 'lns':
            return LNSImprover(time_budget, max_iter, seed)
        if name == 'pbs':
            return PBSSolver(time_budget, node_budget)
        if name == 'icbs':
            return ICBSSolver(time_budget, node_budget)
        if name == 'adaptive':
            root = self.records_dir
            return AdaptiveSolver(
                time_budget, seed, max_iter, node_budget, self.map_name,
                root / 'run_log.jsonl',
            )
        raise ValueError(f"unknown solver: {name}")
