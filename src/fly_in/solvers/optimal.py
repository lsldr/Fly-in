"""Proof-required minimum-makespan planning over the native ICBS engine."""

from fly_in.realm.network import DroneNetwork
from fly_in.solvers.base import SearchBudget, SearchTimeout, Solution
from fly_in.solvers.icbs import ICBSSolver


class UnprovedPlanningError(SearchTimeout):
    """Expose interrupted proof as an error, with a diagnostic incumbent."""

    def __init__(
        self, status: str, incumbent: Solution | None = None,
    ) -> None:
        """Retain feasible diagnostic data without authorizing playback."""
        self.status = status
        self.incumbent = incumbent
        super().__init__(
            f'planning unproved: {status}; minimum makespan proof was not '
            'completed (select an explicit budgeted solver for a feasible run)'
        )


class OptimalSolver(ICBSSolver):
    """Require a legal minimum-turn certificate before returning success.

    The shared ICBS engine bounds paths by a feasible incumbent horizon,
    retains complete capacity-plus-one splitting, and orders OPEN by maximum
    earliest arrival. Branches, bypasses and duplicate constraint sets
    preserve all improving solution families. Limits are optional and apply
    throughout seeding, analysis and search. Secondary optimality is unclaimed.
    """

    def __init__(
        self, time_budget: float | None = None,
        node_budget: int | None = None,
    ) -> None:
        """Default to unlimited proof work while accepting explicit cutoffs."""
        super().__init__(time_budget, node_budget)
        self.proof_required = True
        self.unproved: UnprovedPlanningError | None = None

    def solve(
        self, network: DroneNetwork, drone_ids: list[str],
        initial: Solution | None = None, budget: SearchBudget | None = None,
    ) -> Solution:
        """Return a proved schedule or raise a distinct unproved error.

        Explicit cancellation propagates as PlanningCancelled. An unreachable
        topology remains ValueError. A diagnostic incumbent is never installed
        as a prepared solution after interruption.
        """
        self.unproved = None
        self.solution = None
        try:
            result = super().solve(network, drone_ids, initial, budget)
        except SearchTimeout as error:
            self.unproved = UnprovedPlanningError('timeout')
            raise self.unproved from error
        if not result.optimal:
            self.unproved = UnprovedPlanningError(result.status, result)
            raise self.unproved
        return result
