"""Rung 0: deterministic prioritized planning through shared space-time A*."""

from fly_in.realm.network import DroneNetwork
from fly_in.solvers.base import (
    PlanningCancelled as PlanningCancelled,
    ReservationTable, ScheduledSolver, SearchBudget, SearchStats, Solution,
    SpaceTimeAStar, TimedPath, AgentOrder, SolutionValidator,
)

__all__ = ['PrioritizedPlanner', 'PriorityBasedSolver', 'PlanningCancelled',
           'SeedPlanner']


class PrioritizedPlanner(ScheduledSolver):
    """Feasible floor planner reserving drones in numeric ID order.

    Attributes:
        horizon (int | None): Optional fixed latest arrival turn.
    """

    def __init__(self, horizon: int | None = None) -> None:
        """Optionally impose an explicit horizon, otherwise grow safely.

        Args:
            horizon (int | None): Positive arrival limit, or None to choose a
                constructive bound that grows with prior reservations.

        Raises:
            ValueError: If an explicit horizon is less than one turn.
        """
        super().__init__()
        if horizon is not None and horizon < 1:
            raise ValueError('planning horizon must be positive')
        self.horizon = horizon

    def solve(
        self, network: DroneNetwork, drone_ids: list[str],
        initial: Solution | None = None, budget: SearchBudget | None = None,
    ) -> Solution:
        """Search all agents using a cached exact distance heuristic.

        Args:
            network (DroneNetwork): Topology to route.
            drone_ids (list[str]): Fleet identifiers to schedule.
            initial (Solution | None): Unused incumbent accepted by the solver
                interface.
            budget (SearchBudget | None): Optional cancellation/deadline.

        Returns:
            Solution: Feasible paths in deterministic reservation order.

        Raises:
            ValueError: If the goal is unreachable.
            RuntimeError: If a fixed horizon cannot fit a route.
            PlanningCancelled: If the budget requests cancellation.
            SearchTimeout: If the budget deadline elapses.
        """
        limits = budget or SearchBudget()
        engine = SpaceTimeAStar(network, limits)
        if network.start.name not in engine.distances:
            raise ValueError('no accessible route from start to end')
        duration = engine.distances[network.start.name]
        paths = self._build_schedule(network, drone_ids, engine, limits,
                                     duration)
        return Solution(paths, stats=SearchStats(expanded=engine.expanded))

    def _build_schedule(
        self, network: DroneNetwork, drone_ids: list[str],
        engine: SpaceTimeAStar, budget: SearchBudget, duration: int,
    ) -> dict[str, TimedPath]:
        """Reserve paths, retaining a constructive serial completion bound.

        Args:
            network (DroneNetwork): Topology being planned.
            drone_ids (list[str]): Drone identifiers to schedule.
            engine (SpaceTimeAStar): Shared low-level search engine.
            budget (SearchBudget): Cooperative search limits.
            duration (int): Shortest route duration used in the horizon bound.

        Returns:
            dict[str, TimedPath]: One reserved path per requested drone.

        Raises:
            RuntimeError: If no route fits the selected horizon.
            PlanningCancelled: If the budget requests cancellation.
            SearchTimeout: If the budget deadline elapses.
        """
        paths: dict[str, TimedPath] = {}
        reservations = ReservationTable(network)
        latest = 0
        initial_horizon = max(100, 2 * (len(network.zones)+len(drone_ids)+1))
        for agent in sorted(drone_ids, key=AgentOrder.key):
            budget.check()
            # Waiting at the unlimited start until prior paths finish is a
            # constructive fallback. The roadmap horizon is an initial bound,
            # not a reason to reject a connected graph with a large fleet.
            horizon = (self.horizon if self.horizon is not None
                       else max(initial_horizon, latest+duration))
            path = engine.find_path(agent, reservations, horizon=horizon)
            if path is None:
                raise RuntimeError(
                    f'planning horizon exhausted: no route within {horizon} '
                    'turns'
                )
            paths[agent] = path
            latest = max(latest, path.arrival)
            reservations.reserve(path)
        return paths


class PriorityBasedSolver(PrioritizedPlanner):
    """Backward-compatible public name for the Rung 0 planner."""


class SeedPlanner:
    """Feasible incumbent provider using an injected floor planner.

    Attributes:
        floor_planner (ScheduledSolver): Planner used when no incumbent exists.
    """

    def __init__(self, floor_planner: ScheduledSolver) -> None:
        """Bind the planner used when no incumbent has been supplied.

        Args:
            floor_planner (ScheduledSolver): Planner that constructs the floor.
        """
        self.floor_planner = floor_planner

    def obtain(
        self, network: DroneNetwork, drone_ids: list[str],
        initial: Solution | None, budget: SearchBudget,
    ) -> Solution:
        """Validate/reuse an incumbent or prepare a budgeted feasible floor.

        Callers implement soft local improvement deadlines by passing the
        parent budget here. A valid supplied floor remains available as
        diagnostic data after a deadline; cancellation always takes priority.

        Args:
            network (DroneNetwork): Topology to plan.
            drone_ids (list[str]): Exact fleet identifiers.
            initial (Solution | None): Optional candidate floor to validate.
            budget (SearchBudget): Parent interruption controls.

        Returns:
            Solution: Validated incumbent or a newly planned feasible floor.

        Raises:
            ValueError: If the supplied or generated floor is invalid.
            PlanningCancelled: If the parent cancellation probe fires.
            SearchTimeout: If planning exceeds its deadline.
        """
        SearchBudget(cancelled=budget.cancelled).check()
        if initial is not None:
            SolutionValidator(network).validate(
                initial, drone_ids, SearchBudget(cancelled=budget.cancelled),
            )
            return initial
        budget.check()
        result = self.floor_planner.solve(
            network, drone_ids, budget=budget,
        )
        SolutionValidator(network).validate(result, drone_ids, budget)
        return result
