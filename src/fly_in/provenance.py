"""Immutable planning evidence, independent of simulation completion."""

from dataclasses import asdict, dataclass

from fly_in.solvers.base import Solution
from fly_in.solvers.protocol import Solver


@dataclass(frozen=True)
class PlanningProvenance:
    """Describe a planning result without treating delivery as a proof.

    Attributes:
        optimal_makespan (bool): Whether the planner proved the makespan.
        status (str): Planning status, or ``not_prepared`` without a result.
        proof_reason (str | None): Explanation attached to the planner result.
        lower_bound (int | None): Proven makespan lower bound, when known.
        achieved_makespan (int | None): Makespan of an available solution.
    """

    optimal_makespan: bool = False
    status: str = 'not_prepared'
    proof_reason: str | None = None
    lower_bound: int | None = None
    achieved_makespan: int | None = None

    @classmethod
    def capture(
        cls, solver: Solver | None, error: Exception | None = None,
    ) -> 'PlanningProvenance':
        """Snapshot a solution or an interrupted planner's diagnostic floor.

        An interrupted incumbent remains diagnostic evidence even when its
        paths are legal. It cannot supply an optimal-success flag or enable
        playback. Unscheduled strategies have no planning certificate.

        Args:
            solver (Solver | None): Planner whose solution or interruption is
                inspected.
            error (Exception | None): Failure raised during planning, if any.

        Returns:
            PlanningProvenance: Immutable snapshot of planning evidence.
        """
        interrupted = getattr(solver, 'unproved', None) or error
        candidate = getattr(solver, 'solution', None)
        if interrupted is not None:
            incumbent = getattr(interrupted, 'incumbent', None)
            if isinstance(incumbent, Solution):
                candidate = incumbent
            status = getattr(interrupted, 'status', 'failed')
            bound = getattr(interrupted, 'lower_bound', None)
            if bound is None and isinstance(candidate, Solution):
                bound = candidate.lower_bound
            return cls(
                optimal_makespan=False,
                status=str(status),
                proof_reason=str(interrupted),
                lower_bound=bound if isinstance(bound, int) else None,
                achieved_makespan=(
                    candidate.makespan if isinstance(candidate, Solution)
                    else None
                ),
            )
        if not isinstance(candidate, Solution):
            return cls()
        return cls(
            optimal_makespan=candidate.optimal,
            status=candidate.status,
            proof_reason=candidate.reason,
            lower_bound=candidate.lower_bound,
            achieved_makespan=candidate.makespan,
        )

    def record(self) -> dict[str, object]:
        """Return serializable planning fields without execution status.

        Returns:
            dict[str, object]: Field mapping suitable for JSON encoding.
        """
        return asdict(self)
