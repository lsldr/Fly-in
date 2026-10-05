"""Default optimum and independent physical replay on adversarial fixtures."""

import pytest

from joint_oracle import exact_makespan
from optimality_cases import (
    AnalyticCase, CATALOG_TURNS, EXTRA, HARD_CASE, LARGE_CASES, ROOT,
    SMALL_NAMES, large_map,
)
from schedule_replay import replay
from test_simulation_regressions import Script
from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solver_factory import DEFAULT_SOLVER, SolverFactory
from fly_in.solvers.base import Solution, TimedMove, TimedPath
from fly_in.solvers.optimal import OptimalSolver, UnprovedPlanningError


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_adversarial_map_matches_joint_oracle(name: str) -> None:
    """Use exhaustive legal-state enumeration, then replay actual rows."""
    config = Parser().parse(EXTRA / name)
    expected = exact_makespan(
        DroneNetwork.from_config(config), config.nb_drones,
    )
    solver = SolverFactory().create(DEFAULT_SOLVER)
    simulation = Simulation.from_config(config, solver)
    result = simulation.run(max_turns=30)
    assert result.turns == expected
    assert isinstance(solver, OptimalSolver) and solver.solution is not None
    assert solver.solution.optimal and solver.solution.reason
    assert solver.solution.status == "optimal"
    independent = replay(config, result.history)
    assert independent.cost == result.metrics.total_path_cost


@pytest.mark.parametrize("case", LARGE_CASES, ids=lambda case: case.filename)
def test_full_envelope_map_attains_independent_cut_bound(
    case: AnalyticCase,
) -> None:
    """One fewer turn cannot fit the fleet; legal replay attains the bound."""
    config = Parser().parse(EXTRA / case.filename)
    assert (config.nb_drones, len(config.zones) + 2,
            len(config.connections)) == (100, 200, 300)
    expected = case.minimum(config.nb_drones)
    assert case.deliveries_by(expected - 1) < config.nb_drones
    assert case.deliveries_by(expected) >= config.nb_drones
    solver = SolverFactory().create(DEFAULT_SOLVER)
    simulation = Simulation.from_config(config, solver)
    result = simulation.run(max_turns=expected + 1)
    assert result.turns == expected
    assert isinstance(solver, OptimalSolver) and solver.solution is not None
    assert solver.solution.optimal and solver.solution.status == "optimal"
    assert solver.solution.lower_bound is not None
    assert solver.solution.lower_bound <= expected
    independent = replay(config, result.history)
    assert independent.cost == result.metrics.total_path_cost


@pytest.mark.parametrize("case", LARGE_CASES, ids=lambda case: case.filename)
def test_stress_map_retains_its_analytic_topology(case: AnalyticCase) -> None:
    """Fixture edits must not silently invalidate their mathematical proofs."""
    text = (EXTRA / case.filename).read_text(encoding="utf-8")
    assert text == large_map(case)


@pytest.mark.parametrize("name,minimum", sorted(CATALOG_TURNS.items()))
def test_catalog_minimum_and_output_independently_replayed(
    name: str, minimum: int,
) -> None:
    """Guard documented minima and physical legality for all supplied maps."""
    config = Parser().parse(ROOT / "maps" / name)
    solver = SolverFactory().create(DEFAULT_SOLVER)
    result = Simulation.from_config(config, solver).run(max_turns=minimum + 1)
    assert result.turns == minimum
    assert isinstance(solver, OptimalSolver) and solver.solution is not None
    # These maps attain the reviewed necessary resource-window flow bound.
    assert solver.solution.lower_bound == minimum
    assert solver.solution.optimal and solver.solution.status == "optimal"
    independent = replay(config, result.history)
    assert independent.cost == result.metrics.total_path_cost


def test_extra_catalog_has_only_reviewed_fixtures() -> None:
    """Require every extra map to have either an oracle or analytic test."""
    covered = {*SMALL_NAMES, *(case.filename for case in LARGE_CASES),
               HARD_CASE.filename}
    assert {path.name for path in EXTRA.glob("*.txt")} == covered


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_reversing_declarations_preserves_optimum(name: str) -> None:
    """Tie-breaking must not turn declaration order into a worse schedule."""
    text = (EXTRA / name).read_text(encoding="utf-8")
    rows = text.splitlines()
    headers = [row for row in rows
               if not row.startswith(("hub:", "connection:"))]
    hubs = [row for row in rows if row.startswith("hub:")]
    links = [row for row in rows if row.startswith("connection:")]
    config = Parser().parse_text("\n".join(headers + hubs[::-1] + links[::-1]))
    expected = exact_makespan(
        DroneNetwork.from_config(config), config.nb_drones,
    )
    result = Simulation.from_config(
        config, SolverFactory().create(DEFAULT_SOLVER),
    ).run(max_turns=30)
    assert result.turns == expected
    replay(config, result.history)


def crossing_witness() -> Solution:
    """Construct two disjoint three-step routes for fifty drones apiece."""
    paths = {}
    for index in range(1, 101):
        agent = f"D{index}"
        route = ("s", "c", "b", "g") if index <= 50 else (
            "s", "a", "d", "g",
        )
        paths[agent] = TimedPath(
            agent, tuple((name, turn) for turn, name in enumerate(route)),
            tuple(TimedMove(first, second, "-".join(sorted((first, second))),
                            turn, turn + 1)
                  for turn, (first, second)
                  in enumerate(zip(route, route[1:]))),
        )
    return Solution(paths)


def test_mass_priority_crossing_has_constructive_three_turn_optimum() -> None:
    """Prove the stress optimum with a legal witness and necessary latency."""
    text = (EXTRA / HARD_CASE.filename).read_text(encoding="utf-8")
    assert text == large_map(HARD_CASE)
    config = Parser().parse_text(text)
    assert (config.nb_drones, len(config.zones) + 2,
            len(config.connections)) == (100, 200, 300)
    witness = crossing_witness()
    assert witness.makespan == HARD_CASE.minimum() == 3
    assert HARD_CASE.deliveries_by(2) == 0
    result = Simulation.from_config(config, Script(witness.schedule())).run()
    assert result.turns == 3 and replay(config, result.history).cost == 300
    # This proves seed difficulty is distinct from checking a good incumbent.
    solved = OptimalSolver().solve(
        DroneNetwork.from_config(config), list(witness.paths), initial=witness,
    )
    assert solved.optimal and solved.makespan == solved.lower_bound == 3


def test_mass_priority_crossing_cutoff_keeps_proof_honest() -> None:
    """Wide capacity splits must keep proof interruption honest and atomic."""
    config = Parser().parse(EXTRA / HARD_CASE.filename)
    solver = OptimalSolver(node_budget=1)
    simulation = Simulation.from_config(config, solver)
    try:
        solver.prepare(simulation.network, simulation.drones)
    except UnprovedPlanningError as error:
        assert error.status == "node_limit"
        assert error.incumbent is not None
        assert not error.incumbent.optimal
        assert error.incumbent.lower_bound == 3
        assert error.incumbent.makespan >= 3
        assert solver.solution is None
    else:
        assert solver.solution is not None
        assert solver.solution.optimal and solver.solution.makespan == 3
    assert simulation.turn == 0 and simulation.history == []
