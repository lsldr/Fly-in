"""Supplied-map target gate across every feasible solver rung."""

from pathlib import Path

import pytest

from fly_in.benchmark import TARGETS
from fly_in.solver_factory import SolverFactory
from fly_in.parser import Parser
from fly_in.simulation import Simulation
from fly_in.solvers.base import ScheduledSolver
from fly_in.solvers.protocol import PreparableSolver

ROOT = Path(__file__).resolve().parents[1] / 'maps'


@pytest.mark.parametrize('relative,target', sorted(TARGETS.items()))
@pytest.mark.parametrize('strategy', (
    'priority', 'lns', 'pbs', 'icbs', 'adaptive',
))
def test_supplied_target_for_each_rung(
    relative: str, target: int, strategy: str, tmp_path: Path,
) -> None:
    """Execute the complete schedule, including restricted arrival events."""
    path = ROOT / relative
    solver = SolverFactory(relative, tmp_path).create(
        strategy,
        {'time_budget': 0.015, 'max_iter': 20, 'node_budget': 100},
    )
    simulation = Simulation.from_config(Parser().parse(path), solver)
    assert isinstance(solver, PreparableSolver)
    solver.prepare(simulation.network, simulation.drones)
    result = simulation.run()
    assert (result.turns < target if relative.startswith('challenger/')
            else result.turns <= target)
    assert isinstance(solver, ScheduledSolver)
    assert solver.solution is not None
    assert result.metrics.total_path_cost == solver.solution.cost
