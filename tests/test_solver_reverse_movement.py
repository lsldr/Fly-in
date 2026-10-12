"""Reverse travel must earn its movement cost under scheduling constraints."""

from pathlib import Path

import pytest

from fly_in.parser import Parser
from fly_in.simulation import Simulation
from fly_in.solver_factory import SolverFactory
from fly_in.solvers.base import Constraint, ScheduledSolver, SpaceTimeAStar
from fly_in.solvers.prioritized import PriorityBasedSolver


@pytest.mark.parametrize('strategy', (
    'priority', 'lns', 'pbs', 'icbs', 'adaptive', 'optimal',
))
@pytest.mark.parametrize('kind', ('normal', 'restricted', 'priority'))
def test_prepared_schedules_wait_instead_of_backtracking(
    strategy: str, kind: str, tmp_path: Path,
) -> None:
    """Congested corridors and attractive dead ends must not cause loops.

    Args:
        strategy (str): Prepared solver strategy selected for this shared
            regression.
        kind (str): Zone type assigned to the first intermediate hub to vary
            movement duration and priority preference.
        tmp_path (Path): Pytest temporary directory used for solver run logs.
    """
    config = Parser().parse_text(f'''nb_drones: 4
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0 [zone={kind}]
hub: b 2 0 [zone=restricted]
hub: x 1 1 [zone=priority]
connection: s-a
connection: a-b
connection: b-g
connection: a-x
''')
    solver = SolverFactory(records_dir=tmp_path).create(strategy)
    simulation = Simulation.from_config(config, solver)
    assert isinstance(solver, ScheduledSolver)
    solver.prepare(simulation.network, simulation.drones)
    assert solver.solution is not None
    for path in solver.solution.paths.values():
        assert [(move.source, move.destination) for move in path.moves] == [
            ('s', 'a'), ('a', 'b'), ('b', 'g'),
        ]
    result = simulation.run()
    assert result.turns == solver.solution.makespan
    assert all(drone.delivered for drone in simulation.drones)


@pytest.mark.parametrize('must_vacate', (False, True))
def test_reverse_move_only_when_waiting_is_forbidden(
    must_vacate: bool,
) -> None:
    """A forced departure needs a siding; otherwise waiting costs less.

    Args:
        must_vacate (bool): Whether occupancy of the first intermediate hub
            is forbidden at turn two, requiring a detour through the siding
            before returning to the corridor.
    """
    config = Parser().parse_text('''nb_drones: 1
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0
hub: b 2 0
hub: x 1 1 [zone=priority]
connection: s-a
connection: a-b
connection: b-g
connection: a-x
''')
    network = Simulation.from_config(config, PriorityBasedSolver()).network
    constraints = [
        Constraint('D1', 'link', 'a-s', turn) for turn in range(2, 6)
    ] + [Constraint('D1', 'zone', 'b', turn) for turn in (2, 3)]
    if must_vacate:
        constraints.append(Constraint('D1', 'zone', 'a', 2))
    path = SpaceTimeAStar(network).find_path(
        'D1', constraints=constraints, horizon=5,
    )
    assert path is not None and path.arrival == 5
    destinations = [move.destination for move in path.moves]
    assert destinations == (['a', 'x', 'a', 'b', 'g'] if must_vacate
                            else ['a', 'b', 'g'])
    assert path.cost == (5 if must_vacate else 3)
