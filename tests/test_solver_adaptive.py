"""Portfolio fallback, logging, and proof-status checks."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fly_in.parser import Parser
from fly_in.simulation import Simulation
from fly_in.solvers.adaptive import AdaptiveSolver
from fly_in.solvers.base import SearchTimeout
from fly_in.solvers.lns import LNSImprover
from fly_in.solvers.optimal import OptimalSolver
from fly_in.solvers.prioritized import PrioritizedPlanner


MAP = '''nb_drones: 3
start_hub: s 0 0
end_hub: g 2 0
hub: h 1 0
connection: s-h
connection: h-g
'''


def test_budget_zero_keeps_floor_and_logs_one_record(tmp_path: Path) -> None:
    """The soft initial deadline preserves feasibility and honest timing.

    Args:
        tmp_path (Path): Pytest temporary directory used for logs or generated
            input.
    """
    config = Parser().parse_text(MAP)
    path = tmp_path / 'run_log.jsonl'
    solver = AdaptiveSolver(time_budget=0, map_name='tiny', log_path=path)
    simulation = Simulation.from_config(config, solver)
    baseline = PrioritizedPlanner().solve(
        simulation.network, [drone.id for drone in simulation.drones],
    )
    solver.prepare(simulation.network, simulation.drones)
    assert solver.solution is not None
    assert solver.solution.objective <= baseline.objective
    assert simulation.run().turns == solver.solution.makespan
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['map'] == 'tiny'
    assert rows[0]['initial_makespan'] == baseline.makespan
    assert rows[0]['final_path_cost'] <= baseline.cost
    assert rows[0]['elapsed_ms'] >= 0
    assert solver.solution.optimal
    assert solver.solution.status == rows[0]['status'] == 'optimal'
    assert solver.solution.lower_bound == rows[0]['lower_bound'] == 4
    assert solver.solution.reason == rows[0]['proof_reason']


def test_small_instance_attempts_icbs_when_budget_allows(
    tmp_path: Path,
) -> None:
    """The feature policy sends tiny instances through the exact rung.

    Args:
        tmp_path (Path): Pytest temporary directory used for logs or generated
            input.
    """
    solver = AdaptiveSolver(time_budget=0.1,
                            log_path=tmp_path / 'run_log.jsonl')
    simulation = Simulation.from_config(Parser().parse_text(MAP), solver)
    solver.prepare(simulation.network, simulation.drones)
    assert solver.profile is not None
    assert solver.profile.fleet_size == 3
    assert 'lns' in solver.stage_results
    assert 'icbs' in solver.stage_results
    assert solver.solution is not None
    assert solver.solution.optimal
    assert simulation.run().turns == 4


@pytest.mark.parametrize('node_budget', [0, 10000])
def test_ct_proof_and_cutoff_propagate_above_flow_bound(
    tmp_path: Path, node_budget: int,
) -> None:
    """Report the actual ICBS outcome when the flow bound cannot certify it.

    Args:
        tmp_path (Path): Isolated run-log directory.
        node_budget (int): Zero forces a cutoff; 10000 allows the proof.
    """
    solver = AdaptiveSolver(
        time_budget=5, max_iter=0, node_budget=node_budget,
        log_path=tmp_path / 'run.jsonl',
    )
    simulation = Simulation.from_config(
        Parser().parse(Path(__file__).parent / 'fixtures'
                       / 'flow_relaxation_gap.txt'), solver,
    )
    solver.prepare(simulation.network, simulation.drones)
    result = solver.solution
    assert result is not None
    assert result.makespan == simulation.run().turns == 5
    assert result.lower_bound == 4
    assert result.optimal == (node_budget > 0)
    assert result.status == ('optimal' if node_budget else 'node_limit')
    stage = solver.stage_results['icbs']
    log = json.loads(solver.log_path.read_text())
    for row in (stage, log):
        assert row['optimal_makespan'] == result.optimal
        assert row['status'] == result.status
        assert row['proof_reason'] == result.reason
        assert row['lower_bound'] == result.lower_bound
    if node_budget:
        assert result.reason and 'CT' in result.reason


def test_internal_ct_proof_survives_portfolio_timeout(tmp_path: Path) -> None:
    """A later improvement cutoff does not invalidate an existing proof.

    Args:
        tmp_path (Path): Isolated run-log directory.
    """
    simulation = Simulation.from_config(
        Parser().parse(Path(__file__).parent / 'fixtures'
                       / 'flow_relaxation_gap.txt'), OptimalSolver(),
    )
    agents = [drone.id for drone in simulation.drones]
    floor = OptimalSolver().solve(simulation.network, agents)
    solver = AdaptiveSolver(time_budget=5, log_path=tmp_path / 'run.jsonl')
    with patch.object(LNSImprover, 'solve', return_value=floor), patch.object(
        AdaptiveSolver, '_deep_stage', side_effect=SearchTimeout,
    ):
        result = solver.solve(simulation.network, agents)
    assert result.optimal and result.status == 'optimal'
    assert result.lower_bound == floor.lower_bound == 4
    assert result.makespan == floor.makespan == 5
    assert result.reason == floor.reason
