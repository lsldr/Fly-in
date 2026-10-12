"""Proof-required minimum-makespan planning over the native ICBS engine."""

from fly_in.realm.network import DroneNetwork
from fly_in.solvers.base import SearchBudget, SearchTimeout, Solution
from fly_in.solvers.icbs import ICBSSolver


class UnprovedPlanningError(SearchTimeout):
    """Interrupted proof error with an optional diagnostic incumbent.

    Attributes:
        status (str): Search status at interruption or failure to prove.
        incumbent (Solution | None): Feasible diagnostic result, if available.
    """

    def __init__(
        self, status: str, incumbent: Solution | None = None,
    ) -> None:
        """Retain feasible diagnostic data without authorizing playback.

        Args:
            status (str): Reason proof did not complete.
            incumbent (Solution | None): Feasible schedule found so far.
        """
        self.status = status
        self.incumbent = incumbent
        super().__init__(
            f'planning unproved: {status}; minimum makespan proof was not '
            'completed (select an explicit budgeted solver for a feasible run)'
        )


class OptimalSolver(ICBSSolver):
    """Proof-required minimum-makespan solver using the shared ICBS engine.

    The shared ICBS engine bounds paths by a feasible incumbent horizon,
    retains complete capacity-plus-one splitting, and orders OPEN by maximum
    earliest arrival. Branches, bypasses and duplicate constraint sets
    preserve all improving solution families. Limits are optional and apply
    throughout seeding, analysis and search. Secondary optimality is unclaimed.

    Attributes:
        proof_required (bool): Whether incomplete proof raises instead of
            returning a feasible but unproved floor.
        solution (Solution | None): Validated schedule prepared by the solver.
        unproved (UnprovedPlanningError | None): Last proof failure, if any.
    """

    def __init__(
        self, time_budget: float | None = None,
        node_budget: int | None = None,
    ) -> None:
        """Configure proof cutoffs, unlimited by default.

        Args:
            time_budget (float | None): Optional nonnegative proof duration.
            node_budget (int | None): Optional nonnegative CT node limit.

        Raises:
            ValueError: If an explicit time or node limit is invalid.
        """
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

        Args:
            network (DroneNetwork): Topology to plan.
            drone_ids (list[str]): Fleet identifiers to schedule.
            initial (Solution | None): Optional feasible incumbent.
            budget (SearchBudget | None): Parent deadline and cancellation.

        Returns:
            Solution: Schedule with a completed makespan proof.

        Raises:
            UnprovedPlanningError: If search ends without a proof, carrying
                the feasible diagnostic incumbent when available.
            PlanningCancelled: If explicit cancellation is requested.
            ValueError: If the requested topology or incumbent is invalid.
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
