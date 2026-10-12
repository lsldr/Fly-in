"""Rung 1: seeded destroy-and-repair improvement of feasible schedules.

This adapts MAPF-LNS (Li et al., IJCAI 2021) to makespan, then physical
movement cost. The 70/30 neighborhood policy is Fly-In's roadmap heuristic,
not a reproduction of the paper's adaptive, sum-of-costs implementation.
"""

from math import isfinite
from random import Random

from fly_in.realm.network import DroneNetwork
from fly_in.solvers.base import (
    ReservationTable, ScheduledSolver, SearchBudget, SearchStats,
    SearchTimeout, Solution, SpaceTimeAStar, TimedPath, AgentOrder,
)
from fly_in.solvers.prioritized import PrioritizedPlanner, SeedPlanner


class LNSImprover(ScheduledSolver):
    """Feasible-schedule improver using small destroy-and-repair neighborhoods.

    Attributes:
        time_budget (float): Soft improvement budget in seconds.
        max_iter (int): Maximum destroy-and-repair attempts.
        seed (int): Seed used for reproducible neighborhood selection.
    """

    def __init__(
        self, time_budget: float = 0.5, max_iter: int = 200, seed: int = 0,
    ) -> None:
        """Configure a soft floor-inclusive deadline and reproducible RNG.

        Args:
            time_budget (float): Nonnegative improvement duration in seconds.
            max_iter (int): Nonnegative number of repair attempts.
            seed (int): Random seed for neighborhood sampling.

        Raises:
            ValueError: If duration is non-finite/negative or iteration count
                is negative.
        """
        super().__init__()
        if not isfinite(time_budget) or time_budget < 0:
            raise ValueError('time budget must be finite and nonnegative')
        if max_iter < 0:
            raise ValueError('iteration limit must be nonnegative')
        self.time_budget = time_budget
        self.max_iter = max_iter
        self.seed = seed

    def solve(
        self, network: DroneNetwork, drone_ids: list[str],
        initial: Solution | None = None, budget: SearchBudget | None = None,
    ) -> Solution:
        """Improve a feasible schedule by turns, then movement cost.

        The local improvement deadline does not interrupt floor construction.
        When no incumbent is supplied, floor construction uses the parent
        budget, so its deadline can raise SearchTimeout. After a floor exists,
        timeout abandons partial repair and returns the last accepted schedule.

        Args:
            network (DroneNetwork): Topology to plan.
            drone_ids (list[str]): Fleet identifiers to schedule.
            initial (Solution | None): Optional feasible incumbent.
            budget (SearchBudget | None): Parent limits intersected with this
                improver's deadline.

        Returns:
            Solution: Feasible incumbent, with status indicating cutoff.

        Raises:
            ValueError: If the topology is unreachable or the incumbent is
                invalid for the requested fleet.
            PlanningCancelled: If explicit cancellation is requested.
            SearchTimeout: If a parent deadline expires while constructing a
                missing floor.
        """
        limits = (budget or SearchBudget()).child(self.time_budget)
        floor = SeedPlanner(PrioritizedPlanner()).obtain(
            network, drone_ids, initial, budget or SearchBudget(),
        )
        if set(floor.paths) != set(drone_ids):
            raise ValueError('initial solution must cover the requested fleet')
        stats = SearchStats(expanded=0 if initial else floor.stats.expanded)
        best = Solution(dict(floor.paths), stats=stats)
        rng = Random(self.seed)
        engine: SpaceTimeAStar | None = None
        try:
            limits.check()
            if not drone_ids:
                return best
            table = ReservationTable(network)
            for path in best.paths.values():
                limits.check()
                table.reserve(path)
            engine = SpaceTimeAStar(network, limits)
            for _ in range(self.max_iter):
                limits.check()
                agents = self._neighborhood(network, best, rng, limits)
                rng.shuffle(agents)
                stats.iterations += 1
                replacement = self._repair(
                    best, agents, table, engine, limits,
                )
                if replacement is not None:
                    best.paths = replacement
                    stats.accepted += 1
            best.status = 'iteration_limit'
        except SearchTimeout:
            best.status = 'timeout'
            best.reason = 'LNS time budget exhausted; retained incumbent'
        finally:
            if engine is not None:
                stats.expanded += engine.expanded
        return best

    def _neighborhood(
        self, network: DroneNetwork, incumbent: Solution, rng: Random,
        budget: SearchBudget,
    ) -> list[str]:
        """Choose latest-arrival/shared-resource or uniform random agents.

        Args:
            network (DroneNetwork): Topology used to compare path resources.
            incumbent (Solution): Current feasible schedule.
            rng (Random): Generator used for sampling and policy choice.
            budget (SearchBudget): Interruption controls checked per agent.

        Returns:
            list[str]: Distinct drone identifiers selected for repair.

        Raises:
            PlanningCancelled: If the budget requests cancellation.
            SearchTimeout: If the budget deadline elapses.
        """
        agents = sorted(incumbent.paths, key=AgentOrder.key)
        size = rng.randint(min(2, len(agents)), min(5, len(agents)))
        if rng.random() >= 0.7:
            return rng.sample(agents, size)
        latest = max(agents, key=lambda name: incumbent.paths[name].arrival)
        footprint = self._footprint(network, incumbent.paths[latest])
        sharing = []
        others = []
        for agent in agents:
            budget.check()
            if agent == latest:
                continue
            resources = self._footprint(network, incumbent.paths[agent])
            if footprint & resources:
                sharing.append(agent)
            else:
                others.append(agent)
        selected = rng.sample(sharing, min(size - 1, len(sharing)))
        selected.extend(rng.sample(others, size - 1 - len(selected)))
        return [latest, *selected]

    @staticmethod
    def _footprint(
        network: DroneNetwork, path: TimedPath,
    ) -> set[tuple[str, str]]:
        """Identify shared interior hubs and links without endpoint noise.

        Args:
            network (DroneNetwork): Topology used to omit unlimited endpoints.
            path (TimedPath): Path whose claimed resources are summarized.

        Returns:
            set[tuple[str, str]]: Resource-kind and identifier pairs.
        """
        return {(kind, name) for kind, name, _ in path.resources(network)}

    @staticmethod
    def _repair(
        incumbent: Solution, agents: list[str], table: ReservationTable,
        engine: SpaceTimeAStar, budget: SearchBudget,
    ) -> dict[str, TimedPath] | None:
        """Commit improved reservations, otherwise roll back every change.

        Args:
            incumbent (Solution): Schedule whose selected paths are replaced.
            agents (list[str]): Drone identifiers to remove and replan.
            table (ReservationTable): Mutable reservations for all paths.
            engine (SpaceTimeAStar): Search engine for replacement paths.
            budget (SearchBudget): Interruption controls for repair.

        Returns:
            dict[str, TimedPath] | None: Complete strict improvement, or None
                when repair is infeasible or does not improve the objective.

        Raises:
            PlanningCancelled: If the budget requests cancellation.
            SearchTimeout: If the deadline elapses; reservations are restored
                before propagation.
        """
        original = [incumbent.paths[agent] for agent in agents]
        for path in original:
            table.unreserve(path)
        repaired: list[TimedPath] = []
        accepted = False
        try:
            for agent in agents:
                budget.check()
                replacement = engine.find_path(
                    agent, table, horizon=incumbent.makespan, budget=budget,
                )
                if replacement is None:
                    return None
                table.reserve(replacement)
                repaired.append(replacement)
            paths = dict(incumbent.paths)
            paths.update((path.agent, path) for path in repaired)
            if Solution(paths).objective < incumbent.objective:
                accepted = True
                return paths
            return None
        finally:
            if not accepted:
                for path in repaired:
                    table.unreserve(path)
                for path in original:
                    table.reserve(path)
