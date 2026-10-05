"""Portfolio fallback, logging, and proof-status checks."""

import json
from pathlib import Path

from fly_in.parser import Parser
from fly_in.simulation import Simulation
from fly_in.solvers.adaptive import AdaptiveSolver
from fly_in.solvers.prioritized import PrioritizedPlanner


MAP = '''nb_drones: 3
start_hub: s 0 0
end_hub: g 2 0
hub: h 1 0
connection: s-h
connection: h-g
'''


def test_budget_zero_keeps_floor_and_logs_one_record(tmp_path: Path) -> None:
    """The soft initial deadline preserves feasibility and honest timing."""
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


def test_small_instance_attempts_icbs_when_budget_allows(
    tmp_path: Path,
) -> None:
    """The feature policy sends tiny instances through the exact rung."""
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
