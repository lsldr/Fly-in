"""Physical incumbent legality and prepared-state lifetime regressions."""

from dataclasses import replace
from pathlib import Path

import pytest

from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solvers.base import (
    ScheduledSolver, Solution, SolutionValidator, TimedPath,
)
from fly_in.solvers.optimal import OptimalSolver, UnprovedPlanningError
from fly_in.solvers.lns import LNSImprover
from fly_in.solvers.pbs import PBSSolver
from fly_in.solvers.adaptive import AdaptiveSolver
from fly_in.solvers.prioritized import PrioritizedPlanner


MAP = """nb_drones: 1
start_hub: s 0 0
end_hub: g 2 0
hub: r 1 0 [zone=restricted]
hub: blocked 1 1 [zone=blocked]
connection: s-r
connection: r-g
connection: s-blocked
connection: blocked-g
"""


@pytest.mark.parametrize('defect', [
    'empty', 'unknown-hub', 'blocked-hub', 'bad-duration', 'bad-link',
    'extra-move', 'reversed-moves', 'duplicate-departure', 'post-delivery',
    'wait-gap', 'negative-time',
])
def test_validator_rejects_inconsistent_physical_paths(defect: str) -> None:
    """Reject timed claims that capacity checks alone cannot establish.

    Args:
        defect (str): Malformed path case selected for validator rejection.
    """
    network = DroneNetwork.from_config(Parser().parse_text(MAP))
    path = PrioritizedPlanner().solve(network, ['D1']).paths['D1']
    states = list(path.states)
    moves = list(path.moves)
    if defect == 'empty':
        states = []
    elif defect in ('unknown-hub', 'blocked-hub'):
        states[1] = ('unknown' if defect == 'unknown-hub' else 'blocked', 2)
    elif defect == 'bad-duration':
        states[1] = ('r', 1)
        moves[0] = replace(moves[0], arrival=1)
    elif defect == 'bad-link':
        moves[0] = replace(moves[0], link='s-g')
    elif defect == 'extra-move':
        moves.append(replace(moves[-1], departure=9, arrival=10))
    elif defect == 'reversed-moves':
        moves.reverse()
    elif defect == 'duplicate-departure':
        moves.append(moves[0])
    elif defect == 'post-delivery':
        states.append(('g', 4))
    elif defect == 'wait-gap':
        states.insert(1, ('s', 2))
    else:
        states[1] = ('r', -1)
    malformed = TimedPath('D1', tuple(states), tuple(moves))
    with pytest.raises(ValueError):
        SolutionValidator(network).validate(Solution({'D1': malformed}),
                                            ['D1'])


def test_failed_repreparation_discards_previous_schedule() -> None:
    """An interrupted second fleet cannot expose a previous ready plan."""
    solver = OptimalSolver()
    first = Simulation.from_config(Parser().parse_text(MAP), solver)
    solver.prepare(first.network, first.drones)
    assert solver.solution is not None and solver.has_pending_actions
    trap = Path(__file__).parent / 'fixtures/prioritized_trap.txt'
    second = Simulation.from_config(Parser().parse(trap), solver)
    solver.node_budget = 0
    with pytest.raises(UnprovedPlanningError):
        solver.prepare(second.network, second.drones)
    assert solver.solution is None
    assert not solver.has_pending_actions
    assert solver.unproved is not None
    assert solver.unproved.incumbent is not None


@pytest.mark.parametrize('strategy', ['lns', 'pbs', 'adaptive'])
def test_heuristics_do_not_trust_a_supplied_optimal_flag(
    strategy: str, tmp_path: Path,
) -> None:
    """Legal suboptimal paths cannot establish a caller's proof claim.

    Args:
        strategy (str): Solver strategy selected for this shared regression.
        tmp_path (Path): Pytest temporary directory used for logs or generated
            input.
    """
    trap = Path(__file__).parent / 'fixtures/prioritized_trap.txt'
    network = DroneNetwork.from_config(Parser().parse(trap))
    initial = PrioritizedPlanner().solve(network, ['D1', 'D2'])
    initial.optimal = True
    initial.status = 'optimal'
    initial.reason = 'untrusted caller flag'
    solver: ScheduledSolver
    if strategy == 'lns':
        solver = LNSImprover(time_budget=0)
    elif strategy == 'pbs':
        solver = PBSSolver(time_budget=0)
    else:
        solver = AdaptiveSolver(
            time_budget=0, log_path=tmp_path / 'log.jsonl',
        )
    result = solver.solve(network, ['D1', 'D2'], initial)
    assert result.makespan == 4
    assert not result.optimal
    assert initial.optimal
