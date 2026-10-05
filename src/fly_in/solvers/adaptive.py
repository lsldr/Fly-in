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
    """Select bounded improvement stages using graph and fleet features."""

    def __init__(
        self, time_budget: float = 1.0, seed: int = 0,
        max_iter: int = 200, node_budget: int = 10000,
        map_name: str = 'in_memory',
        log_path: Path | None = None,
    ) -> None:
        """Set portfolio limits and the append-only JSONL destination."""
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
        """Run the floor, quick LNS, and one feature-selected deeper stage."""
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
        if best.makespan == self.profile.lower_bound:
            best.optimal = True
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
            best.status = 'timeout'
            best.reason = 'portfolio improvement deadline exhausted'
        if best.makespan == self.profile.lower_bound:
            best.optimal = True
            best.status = 'optimal'
            best.reason = 'makespan equals certified flow lower bound'
        elapsed = (perf_counter() - started) * 1000
        self._log_run(network, drone_ids, floor, best, elapsed)
        return best

    @staticmethod
    def _remaining(deadline: float) -> float:
        """Return nonnegative wall-clock seconds remaining."""
        return max(0.0, deadline - perf_counter())

    @staticmethod
    def _stage_row(solution: Solution) -> dict[str, object]:
        """Expose measured outcome and proof status for each attempted rung."""
        return {
            'makespan': solution.makespan,
            'path_cost': solution.cost,
            'status': solution.status,
            'optimal_makespan': solution.optimal,
            'iterations': solution.stats.iterations,
            'expanded': solution.stats.expanded,
            'nodes': solution.stats.nodes,
        }

    def _consider(
        self, incumbent: Solution, candidate: Solution,
        stage: str, label: str,
    ) -> None:
        """Adopt only a strict lexicographic improvement or new proof."""
        self.stage_results[stage] = self._stage_row(candidate)
        incumbent.stats.expanded += candidate.stats.expanded
        incumbent.stats.nodes += candidate.stats.nodes
        incumbent.stats.iterations += candidate.stats.iterations
        incumbent.stats.accepted += candidate.stats.accepted
        if candidate.objective < incumbent.objective:
            incumbent.paths = dict(candidate.paths)
            self.winning_rung = label
        if candidate.optimal and candidate.makespan == incumbent.makespan:
            incumbent.optimal = True
            incumbent.reason = candidate.reason

    def _deep_stage(
        self, remaining: float,
    ) -> tuple[str, str, ICBSSolver | PBSSolver | LNSImprover]:
        """Choose exact search for tiny instances, else PBS or more LNS."""
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
        """Append exactly one compact JSON object after completed planning."""
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
            'lower_bound': profile.lower_bound if profile else None,
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
