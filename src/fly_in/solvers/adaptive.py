"""Budgeted solver portfolio with a feasible floor and structured provenance.

The deadline is soft only while constructing the first valid schedule. Later
stages cooperate with the remaining wall-clock budget and retain the best
fully feasible incumbent. Explicit cancellation always interrupts planning.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

from fly_in.realm.network import DroneNetwork
from fly_in.solvers.base import (
    ScheduledSolver, SearchBudget, SearchStats, SearchTimeout, Solution,
)
from fly_in.solvers.bounds import InstanceProfile, NetworkProfiler
from fly_in.solvers.icbs import ICBSSolver
from fly_in.solvers.lns import LNSImprover
from fly_in.solvers.pbs import PBSSolver
from fly_in.solvers.prioritized import PrioritizedPlanner, SeedPlanner


class AdaptiveSolver(ScheduledSolver):
    """Budgeted portfolio selected from graph and fleet features.

    Attributes:
        time_budget (float): Portfolio duration in seconds.
        seed (int): Base seed for randomized improvement stages.
        max_iter (int): LNS iteration limit per stage.
        node_budget (int): High-level node limit for exact/PBS stages.
        map_name (str): Label recorded for the network.
        log_path (Path): Append-only JSONL run-log path.
        profile (InstanceProfile | None): Most recent network profile.
        winning_rung (str): Label of the stage with the incumbent objective.
        stage_results (dict[str, dict[str, object]]): Per-stage measurements.
    """

    def __init__(
        self, time_budget: float = 1.0, seed: int = 0,
        max_iter: int = 200, node_budget: int = 10000,
        map_name: str = 'in_memory',
        log_path: Path | None = None,
    ) -> None:
        """Set portfolio limits and the append-only JSONL destination.

        Args:
            time_budget (float): Finite, nonnegative portfolio seconds.
            seed (int): Base seed for randomized stages.
            max_iter (int): Nonnegative LNS iteration limit.
            node_budget (int): Nonnegative high-level node limit.
            map_name (str): Name recorded in run provenance.
            log_path (Path | None): JSONL output path; defaults to the records
                directory when omitted.

        Raises:
            ValueError: If the duration is invalid or either search limit is
                negative.
        """
        super().__init__()
        if not isfinite(time_budget) or time_budget < 0:
            raise ValueError('time budget must be finite and nonnegative')
        if max_iter < 0 or node_budget < 0:
            raise ValueError('search limits must be nonnegative')
        self.time_budget = time_budget
        self.seed = seed
        self.max_iter = max_iter
        self.node_budget = node_budget
        self.map_name = map_name
        self.log_path = log_path or Path('records/run_log.jsonl')
        self.profile: InstanceProfile | None = None
        self.winning_rung = 'Rung 0 (prioritized A*)'
        self.stage_results: dict[str, dict[str, object]] = {}

    def solve(
        self, network: DroneNetwork, drone_ids: list[str],
        initial: Solution | None = None, budget: SearchBudget | None = None,
    ) -> Solution:
        """Run the feasible floor, quick LNS, and a feature-selected stage.

        The portfolio deadline is soft while constructing a missing floor,
        which uses the parent budget. Once a floor exists, later timeout keeps
        that incumbent. A result matching the certified lower bound receives
        optimal status even if the improvement stage timed out.

        Args:
            network (DroneNetwork): Topology to plan.
            drone_ids (list[str]): Fleet identifiers to schedule.
            initial (Solution | None): Optional feasible incumbent.
            budget (SearchBudget | None): Parent deadline and cancellation.

        Returns:
            Solution: Best feasible schedule found within the portfolio.

        Raises:
            ValueError: If the topology is unreachable or an incumbent is
                invalid.
            PlanningCancelled: If explicit cancellation is requested.
            SearchTimeout: If a parent deadline expires while constructing a
                missing floor.
            OSError: If the JSONL run log cannot be created or appended.
        """
        started = perf_counter()
        parent = budget or SearchBudget()
        deadline = started + self.time_budget
        if parent.deadline is not None:
            deadline = min(deadline, parent.deadline)
        limits = SearchBudget(deadline, parent.cancelled)
        floor = SeedPlanner(PrioritizedPlanner()).obtain(
            network, drone_ids, initial, budget or SearchBudget(),
        )
        best = Solution(
            dict(floor.paths),
            stats=SearchStats(expanded=0 if initial else floor.stats.expanded),
        )
        self.winning_rung = 'Rung 0 (prioritized A*)'
        self.stage_results = {
            'priority': self._stage_row(floor),
        }
        self.profile = NetworkProfiler(network).profile(len(drone_ids))
        best.lower_bound = self.profile.lower_bound
        if best.makespan == self.profile.lower_bound:
            best.optimal = True
            best.status = 'optimal'
            best.reason = 'makespan equals certified flow lower bound'
        try:
            limits.check()
            quick = LNSImprover(
                time_budget=min(0.150, self._remaining(deadline)),
                max_iter=self.max_iter, seed=self.seed,
            ).solve(network, drone_ids, initial=best, budget=limits)
            self._consider(best, quick, 'lns', 'Rung 1 (MAPF-LNS)')
            limits.check()
            remaining = self._remaining(deadline)
            if remaining > 0:
                stage, label, solver = self._deep_stage(remaining)
                candidate = solver.solve(
                    network, drone_ids, initial=best, budget=limits,
                )
                self._consider(best, candidate, stage, label)
        except SearchTimeout:
            if not best.optimal:
                best.status = 'timeout'
                best.reason = 'portfolio improvement deadline exhausted'
        if not best.optimal and best.makespan == self.profile.lower_bound:
            best.optimal = True
            best.status = 'optimal'
            best.reason = 'makespan equals certified flow lower bound'
        elapsed = (perf_counter() - started) * 1000
        self._log_run(network, drone_ids, floor, best, elapsed)
        return best

    @staticmethod
    def _remaining(deadline: float) -> float:
        """Return nonnegative wall-clock seconds remaining.

        Args:
            deadline (float): Absolute ``perf_counter`` deadline in seconds.

        Returns:
            float: Remaining seconds, clamped at zero.
        """
        return max(0.0, deadline - perf_counter())

    @staticmethod
    def _stage_row(solution: Solution) -> dict[str, object]:
        """Expose measured outcome and proof status for an attempted rung.

        Args:
            solution (Solution): Stage result to summarize.

        Returns:
            dict[str, object]: Stable scalar measurements for JSON logging.
        """
        return {
            'makespan': solution.makespan,
            'path_cost': solution.cost,
            'status': solution.status,
            'optimal_makespan': solution.optimal,
            'proof_reason': solution.reason,
            'lower_bound': solution.lower_bound,
            'iterations': solution.stats.iterations,
            'expanded': solution.stats.expanded,
            'nodes': solution.stats.nodes,
        }

    def _consider(
        self, incumbent: Solution, candidate: Solution,
        stage: str, label: str,
    ) -> None:
        """Adopt only a strict lexicographic improvement or new proof.

        Args:
            incumbent (Solution): Mutable best result updated in place.
            candidate (Solution): Completed result from one stage.
            stage (str): Key under which to record stage measurements.
            label (str): Human-readable name for the winning stage.
        """
        self.stage_results[stage] = self._stage_row(candidate)
        incumbent.stats.expanded += candidate.stats.expanded
        incumbent.stats.nodes += candidate.stats.nodes
        incumbent.stats.iterations += candidate.stats.iterations
        incumbent.stats.accepted += candidate.stats.accepted
        if candidate.objective < incumbent.objective:
            incumbent.paths = dict(candidate.paths)
            self.winning_rung = label
        if candidate.lower_bound is not None:
            incumbent.lower_bound = max(
                incumbent.lower_bound or 0, candidate.lower_bound,
            )
        if candidate.optimal and candidate.makespan == incumbent.makespan:
            incumbent.optimal = True
            incumbent.status = 'optimal'
            incumbent.reason = candidate.reason
        elif not incumbent.optimal:
            incumbent.status = candidate.status
            incumbent.reason = candidate.reason

    def _deep_stage(
        self, remaining: float,
    ) -> tuple[str, str, ICBSSolver | PBSSolver | LNSImprover]:
        """Choose exact search for tiny instances, else PBS or more LNS.

        Args:
            remaining (float): Remaining stage budget in seconds.

        Returns:
            tuple[str, str, ICBSSolver | PBSSolver | LNSImprover]: Stage key,
                display label, and configured solver.
        """
        profile = self.profile
        assert profile is not None
        if profile.fleet_size <= 4 and profile.vertices <= 15:
            return ('icbs', 'Rung 3 (ICBS)', ICBSSolver(
                time_budget=remaining, node_budget=self.node_budget,
            ))
        if profile.bottleneck_capacity <= 2:
            return ('pbs', 'Rung 2 (PBS)', PBSSolver(
                time_budget=remaining, node_budget=self.node_budget,
            ))
        return ('lns_deep', 'Rung 1 (MAPF-LNS)', LNSImprover(
            time_budget=remaining, max_iter=self.max_iter,
            seed=self.seed + 1,
        ))

    def _log_run(
        self, network: DroneNetwork, drone_ids: list[str],
        floor: Solution, final: Solution, elapsed_ms: float,
    ) -> None:
        """Append one compact JSON object after completed planning.

        Args:
            network (DroneNetwork): Network whose features are recorded.
            drone_ids (list[str]): Fleet identifiers used in planning.
            floor (Solution): Initial feasible result before improvements.
            final (Solution): Final incumbent and proof metadata.
            elapsed_ms (float): Total measured planning duration in
                milliseconds.

        Side Effects:
            Creates parent directories as needed and appends one JSON line to
            ``log_path``.

        Raises:
            OSError: If the log directory or file cannot be created or written.
        """
        profile = self.profile
        row = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'map': self.map_name,
            'nb_drones': len(drone_ids),
            'winning_rung': self.winning_rung,
            'initial_makespan': floor.makespan,
            'final_makespan': final.makespan,
            'initial_path_cost': floor.cost,
            'final_path_cost': final.cost,
            'iterations': final.stats.iterations,
            'elapsed_ms': round(elapsed_ms, 3),
            'status': final.status,
            'optimal_makespan': final.optimal,
            'lower_bound': final.lower_bound,
            'proof_reason': final.reason,
            'vertices': len(network.zones),
            'edges': len(network.connections),
            'density': profile.density if profile else None,
            'bottleneck_capacity': (
                profile.bottleneck_capacity if profile else None
            ),
            'restricted_ratio': (
                profile.restricted_ratio if profile else None
            ),
            'shortest_distance': (
                profile.shortest_distance if profile else None
            ),
            'stages': self.stage_results,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
