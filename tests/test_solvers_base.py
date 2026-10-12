"""Shared search checks against the executable simulation contract."""

from pathlib import Path

import pytest

from fly_in.parser import Parser
from fly_in.simulation import Simulation
from fly_in.solvers.base import (
    Constraint, PlanningCancelled, ReservationTable, SearchBudget,
    SpaceTimeAStar, ConflictDetector,
)
from fly_in.solvers.prioritized import PriorityBasedSolver


MAP = '''nb_drones: 2
start_hub: s 0 0
end_hub: g 2 0
hub: r 1 0 [zone=restricted]
connection: s-r [max_link_capacity=1]
connection: r-g
'''


def test_link_release_and_schedule_match_simulation() -> None:
    """Second entry shares first arrival without excess live traffic."""
    config = Parser().parse_text(MAP)
    simulation = Simulation.from_config(config, PriorityBasedSolver())
    solver = simulation.solver
    assert isinstance(solver, PriorityBasedSolver)
    solver.prepare(simulation.network, simulation.drones)
    assert solver.solution is not None
    assert not ConflictDetector(simulation.network).find(solver.solution.paths)
    result = simulation.run()
    assert result.turns == 4
    assert result.history[1] == ['D2-r-s', 'D1-r']


def test_reservations_can_be_removed_without_lingering_claims() -> None:
    """A two-turn transit claims the link only at entry time."""
    simulation = Simulation.from_config(
        Parser().parse_text(MAP), PriorityBasedSolver(),
    )
    network = simulation.network
    engine = SpaceTimeAStar(network)
    first = engine.find_path('D1')
    assert first is not None
    table = ReservationTable(network)
    table.reserve(first)
    link = network.connections['r-s']
    assert not table.can_traverse(link, 1)
    assert table.can_traverse(link, 2)
    assert not table.can_occupy(network.zones['r'], 2)
    table.unreserve(first)
    assert table.can_traverse(link, 1)
    assert table.can_occupy(network.zones['r'], 2)
    with pytest.raises(ValueError, match='unreserved'):
        table.unreserve(first)


def test_negative_constraints_apply_on_actual_entry_and_arrival() -> None:
    """Forbidding transit entry shifts arrival; forbidding arrival does too."""
    network = Simulation.from_config(
        Parser().parse_text(MAP), PriorityBasedSolver(),
    ).network
    engine = SpaceTimeAStar(network)
    free = engine.find_path('D1')
    constrained_link = engine.find_path('D1', constraints=[
        Constraint('D1', 'link', 'r-s', 1),
    ])
    constrained_hub = engine.find_path('D1', constraints=[
        Constraint('D1', 'zone', 'r', 2),
    ])
    assert free is not None and free.arrival == 3
    assert constrained_link is not None and constrained_link.arrival == 4
    assert constrained_hub is not None and constrained_hub.arrival == 4


def test_explicit_cancellation_prevents_search(tmp_path: Path) -> None:
    """Cancellation remains distinct from a bounded infeasible route.

    Args:
        tmp_path (Path): Pytest temporary directory used for logs or generated
            input.
    """
    _ = tmp_path
    network = Simulation.from_config(
        Parser().parse_text(MAP), PriorityBasedSolver(),
    ).network
    with pytest.raises(PlanningCancelled):
        SpaceTimeAStar(network, SearchBudget(
            cancelled=lambda: True,
        ))
