"""Independent version 2 checks for CT search and mixed endpoint costs."""

from itertools import combinations
from pathlib import Path
from random import Random
from typing import Callable

import pytest

from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solver_factory import SOLVER_CHOICES, SolverFactory
from fly_in.solvers.base import SolutionValidator
from fly_in.solvers.bounds import NetworkProfiler
from fly_in.solvers.optimal import OptimalSolver
from joint_oracle import exact_makespan
from schedule_replay import replay
from test_optimal_solver import _serial_seed
from test_simulation_regressions import Script


def mixed_map(seed: int) -> str:
    """Generate tiny connected maps including costly endpoints and decoys."""
    random = Random(seed)
    fleet = 2 + seed % 2
    start_kind = "restricted" if seed % 3 == 0 else "normal"
    end_kind = "restricted" if seed % 2 else "priority"
    rows = [f"nb_drones: {fleet}",
            f"start_hub: s 0 0 [zone={start_kind} max_drones=0]",
            f"end_hub: g 5 0 [zone={end_kind} max_drones=ignored]"]
    for index, name in enumerate(("a", "b", "c", "d"), 1):
        kinds = ["normal", "priority", "restricted"]
        if name in ("c", "d"):
            kinds.append("blocked")
        rows.append(f"hub: {name} {index} 0 [zone={random.choice(kinds)} "
                    f"max_drones={random.randint(1, 3)}]")
    edges = {("s", "a"), ("a", "b"), ("b", "g")}
    for first, second in combinations(("s", "a", "b", "c", "d", "g"), 2):
        if random.random() < 0.25:
            edges.add((first, second))
    for first, second in sorted(edges):
        rows.append(f"connection: {first}-{second} "
                    f"[max_link_capacity={random.randint(1, 3)}]")
    return "\n".join(rows) + "\n"


@pytest.mark.parametrize("seed", range(64), ids=lambda seed: f"seed-{seed}")
@pytest.mark.parametrize("loose_seed", [False, True], ids=["default", "ct"])
def test_mixed_endpoint_costs_match_exact_joint_search(
    seed: int, loose_seed: bool,
    record_property: Callable[[str, object], None],
) -> None:
    """Cross-check optimum, bound and physical execution independently."""
    config = Parser().parse_text(mixed_map(seed))
    network = DroneNetwork.from_config(config)
    expected = exact_makespan(network, config.nb_drones)
    lower = NetworkProfiler(network).profile(config.nb_drones).lower_bound
    assert lower <= expected
    initial = _serial_seed(network, config.nb_drones) if loose_seed else None
    solver = OptimalSolver()
    agents = [f"D{index}" for index in range(1, config.nb_drones + 1)]
    solution = solver.solve(network, agents, initial=initial)
    assert solution.optimal and solution.status == "optimal"
    assert solution.reason and solution.makespan == expected
    SolutionValidator(network).validate(solution, agents)
    result = Simulation.from_config(config, Script(solution.schedule())).run(
        max_turns=expected + 1,
    )
    assert result.turns == expected
    assert replay(config, result.history).cost == solution.cost
    record_property("ct_nodes", solution.stats.nodes)
    record_property("bypasses", solution.stats.bypasses)
    if initial is not None and initial.makespan > expected:
        assert solution.stats.nodes > 0


@pytest.mark.parametrize("strategy", SOLVER_CHOICES)
@pytest.mark.parametrize("restricted_goal", [False, True])
def test_every_strategy_prefers_priority_on_equal_physical_routes(
    strategy: str, restricted_goal: bool, tmp_path: Path,
) -> None:
    """Priority choice survives name ordering and a two-turn final entry."""
    goal_kind = "restricted" if restricted_goal else "normal"
    config = Parser().parse_text(f"""nb_drones: 1
start_hub: s 0 0
end_hub: g 2 0 [zone={goal_kind}]
hub: a_normal 1 0
hub: z_priority 1 1 [zone=priority]
connection: s-a_normal
connection: a_normal-g
connection: s-z_priority
connection: z_priority-g
""")
    solver = SolverFactory(records_dir=tmp_path).create(strategy)
    result = Simulation.from_config(config, solver).run(max_turns=5)
    assert result.history[0] == ["D1-z_priority"]
    assert result.turns == (3 if restricted_goal else 2)
    replay(config, result.history)


@pytest.mark.parametrize("both_restricted", [False, True])
def test_connection_release_allows_reverse_departure_with_forced_arrival(
    both_restricted: bool,
) -> None:
    """The released link supports either direction on the same turn."""
    kind = "restricted" if both_restricted else "normal"
    config = Parser().parse_text(f"""nb_drones: 2
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0 [zone={kind}]
hub: b 2 0 [zone=restricted]
connection: s-a
connection: a-b
connection: a-g
connection: b-g
""")
    rows = ([['D1-a-s'], ['D2-a-s'], ['D1-a-b'], ['D2-a-b'],
             ['D1-a-b'], ['D2-g'], ['D1-g']] if both_restricted else
            [['D1-a'], ['D1-a-b', 'D2-a'], ['D2-a-b'], ['D1-a'],
             ['D1-g', 'D2-g']])
    result = Simulation.from_config(config, Script(rows)).run(max_turns=7)
    assert result.turns == (7 if both_restricted else 5)
    replay(config, result.history)


def test_default_horizon_handles_two_hundred_hubs_and_long_restricted_chain(
) -> None:
    """Latency beyond 100 turns cannot be mistaken for unreachable input."""
    names = ["s", *[f"r{index}" for index in range(198)], "g"]
    rows = ["nb_drones: 17", "start_hub: s 0 0", "end_hub: g 199 0"]
    rows += [f"hub: {name} {index} 0 [zone=restricted]"
             for index, name in enumerate(names[1:-1], 1)]
    rows += [f"connection: {first}-{second}"
             for first, second in zip(names, names[1:])]
    config = Parser().parse_text("\n".join(rows))
    # 198 two-turn entries + final normal entry + 16 pipeline followers.
    minimum = 198 * 2 + 1 + 16
    solver = OptimalSolver()
    result = Simulation.from_config(config, solver).run(max_turns=minimum + 1)
    assert result.turns == minimum
    assert solver.solution is not None and solver.solution.optimal
    assert replay(config, result.history).cost == 17 * 397


def test_default_fleet_size_has_no_challenger_or_stress_envelope_cap() -> None:
    """Fleet size may exceed 100 with ignored endpoint occupancy metadata."""
    config = Parser().parse_text("""nb_drones: 257
start_hub: s 0 0 [max_drones=1]
end_hub: g 1 0 [zone=restricted max_drones=0]
connection: s-g [max_link_capacity=7]
""")
    solver = OptimalSolver()
    result = Simulation.from_config(config, solver).run(max_turns=39)
    assert result.turns == 38  # ceil(257 / 7) departures + one transit turn.
    assert solver.solution is not None and solver.solution.optimal
    assert len(replay(config, result.history).arrivals) == 257
