"""LNS feasibility, reproducibility, and transactional repair coverage."""

from collections import Counter
from time import perf_counter
from typing import Literal

import pytest

from fly_in.models import (
    EdgeZoneSpec, MapConfig, RegZoneSpec, RouteSpec, ZoneType,
)
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solvers import lns
from fly_in.solvers.base import (
    Constraint, PlanningCancelled, ReservationTable, SearchBudget,
    SearchTimeout, Solution, SpaceTimeAStar, TimedMove, TimedPath,
    ConflictDetector,
)
from fly_in.solvers.lns import LNSImprover
from fly_in.solvers.prioritized import PrioritizedPlanner


def _config(fleet: int = 4) -> MapConfig:
    """Return a bottleneck with a restricted-entry pipeline and alternate.

    Args:
        fleet (int): Number of drones in the generated map configuration.

    Returns:
        MapConfig: Bottleneck fixture with the requested fleet size.
    """
    return MapConfig(
        nb_drones=fleet,
        start=EdgeZoneSpec(name='s', x=0, y=0),
        end=EdgeZoneSpec(name='g', x=3, y=0),
        zones=[
            RegZoneSpec(name='r', x=1, y=0,
                        zone_type=ZoneType.RESTRICTED),
            RegZoneSpec(name='a', x=1, y=1),
            RegZoneSpec(name='b', x=2, y=1),
        ],
        connections=[RouteSpec(from_hub=a, to_hub=b)
                     for a, b in [('s', 'r'), ('r', 'g'), ('s', 'a'),
                                  ('a', 'b'), ('b', 'g')]],
    )


def _fleet(fleet: int = 4) -> list[str]:
    """Return stable conventional IDs for a small fixture.

    Args:
        fleet (int): Number of drone identifiers to generate.

    Returns:
        list[str]: Stable identifiers D1 through the requested fleet size.
    """
    return [f'D{i}' for i in range(1, fleet + 1)]


def _delayed(solution: Solution, turns: int = 2) -> Solution:
    """Make a feasible but slower incumbent by delaying all departures.

    Args:
        solution (Solution): Feasible schedule whose paths are delayed.
        turns (int): Number of initial wait turns added to every path.

    Returns:
        Solution: Equivalent feasible paths with delayed departures.
    """
    paths = {}
    for agent, path in solution.paths.items():
        waits = tuple(('s', turn) for turn in range(turns))
        states = tuple((name, turn + turns) for name, turn in path.states)
        moves = tuple(TimedMove(
            move.source, move.destination, move.link,
            move.departure + turns, move.arrival + turns,
        ) for move in path.moves)
        paths[agent] = TimedPath(agent, waits + states, moves)
    return Solution(paths)


def test_fixed_seed_and_iteration_limit_are_reproducible() -> None:
    """Wall-clock slack leaves the seeded finite search deterministic."""
    network = DroneNetwork.from_config(_config())
    initial = _delayed(PrioritizedPlanner().solve(network, _fleet()))
    first = LNSImprover(60, 20, seed=7).solve(network, _fleet(), initial)
    second = LNSImprover(60, 20, seed=7).solve(network, _fleet(), initial)
    assert first.paths == second.paths
    assert first.stats == second.stats
    assert first.stats.iterations == 20
    assert first.status == 'iteration_limit'
    assert first.stats.accepted > 0
    assert first.objective < initial.objective
    assert initial.stats.iterations == 0
    assert not ConflictDetector(network).find(first.paths)


@pytest.mark.parametrize('seed', range(5))
def test_improvement_never_worsens_the_floor(seed: int) -> None:
    """Both selection heuristics preserve the incumbent's objective.

    Args:
        seed (int): Fixed random seed used to generate this map.
    """
    network = DroneNetwork.from_config(_config(7))
    initial = PrioritizedPlanner().solve(network, _fleet(7))
    original_paths = dict(initial.paths)
    result = LNSImprover(60, 12, seed).solve(network, _fleet(7), initial)
    assert result.objective <= initial.objective
    assert initial.paths == original_paths
    assert result.paths is not initial.paths
    assert not ConflictDetector(network).find(result.paths)


def test_single_drone_and_simulation_adapter() -> None:
    """A one-agent neighborhood and restricted arrivals replay legally."""
    solver = LNSImprover(60, 3)
    simulation = Simulation.from_config(_config(1), solver)
    result = simulation.run()
    assert solver.solution is not None
    assert solver.solution.makespan == result.turns
    assert solver.solution.cost == result.metrics.total_path_cost
    assert all(drone.delivered for drone in simulation.drones)
    assert solver.solution.stats.iterations == 3


def test_zero_budget_finishes_floor_then_returns_timeout() -> None:
    """The soft deadline cannot discard an initial feasible schedule."""
    network = DroneNetwork.from_config(_config())
    floor = PrioritizedPlanner().solve(network, _fleet())
    result = LNSImprover(0).solve(network, _fleet())
    assert result.paths == floor.paths
    assert result.status == 'timeout'
    assert result.stats.iterations == 0
    assert result.stats.expanded > 0


def test_parent_deadline_caps_the_improvement_stage() -> None:
    """An expired parent budget wins even if the local budget is long."""
    network = DroneNetwork.from_config(_config())
    initial = PrioritizedPlanner().solve(network, _fleet())
    result = LNSImprover(60).solve(
        network, _fleet(), initial,
        SearchBudget(deadline=perf_counter() - 1),
    )
    assert result.paths == initial.paths
    assert result.status == 'timeout'
    assert result.stats.iterations == result.stats.expanded == 0


def test_explicit_cancellation_is_not_a_timeout() -> None:
    """Floor construction remains cancellable even with no time left."""
    network = DroneNetwork.from_config(_config())
    with pytest.raises(PlanningCancelled):
        LNSImprover(0).solve(
            network, _fleet(), budget=SearchBudget(cancelled=lambda: True),
        )


@pytest.mark.parametrize('failure', ['infeasible', 'timeout', 'cancelled'])
def test_partial_repairs_restore_all_reservations(
    monkeypatch: pytest.MonkeyPatch,
    failure: Literal['infeasible', 'timeout', 'cancelled'],
) -> None:
    """Undo newly reserved routes before restoring destroyed old paths.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to patch search
            behavior during the test.
        failure (Literal['infeasible', 'timeout', 'cancelled']): Injected
            repair outcome: infeasible, timeout, or cancellation.
    """
    network = DroneNetwork.from_config(_config(2))
    initial = PrioritizedPlanner().solve(network, _fleet(2))
    captured: list[ReservationTable] = []

    class CapturingTable(ReservationTable):
        """Expose the transient table solely for rollback assertions."""

        def __init__(self, graph: DroneNetwork) -> None:
            """Capture each reservation table created by the patched solver.

            Args:
                graph (DroneNetwork): Topology for the captured table.
            """
            super().__init__(graph)
            captured.append(self)

    monkeypatch.setattr(lns, 'ReservationTable', CapturingTable)
    original_search = SpaceTimeAStar.find_path
    calls = 0

    def interrupted_search(
        engine: SpaceTimeAStar, agent: str,
        reservations: ReservationTable | None = None,
        constraints: tuple[Constraint, ...] = (), horizon: int = 100,
        budget: SearchBudget | None = None,
    ) -> TimedPath | None:
        """Complete one repair, then fail before the second finishes.

        Args:
            engine (SpaceTimeAStar): Engine whose path search is intercepted.
            agent (str): Drone identifier passed to the path search.
            reservations (ReservationTable | None): Existing claims for the
                path search.
            constraints (tuple[Constraint, ...]): Forbidden resource uses for
                the path search.
            horizon (int): Latest arrival turn allowed by the path search.
            budget (SearchBudget | None): Cooperative limits for path search.

        Returns:
            TimedPath | None: Original result, or None for injected
                infeasibility.

        Raises:
            SearchTimeout: On the injected timeout case.
            PlanningCancelled: On the injected cancellation case.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == 'timeout':
                raise SearchTimeout('injected timeout')
            if failure == 'cancelled':
                raise PlanningCancelled('injected cancellation')
            return None
        return original_search(engine, agent, reservations, constraints,
                               horizon, budget)

    monkeypatch.setattr(SpaceTimeAStar, 'find_path', interrupted_search)
    solver = LNSImprover(60, 1)
    if failure == 'cancelled':
        with pytest.raises(PlanningCancelled):
            solver.solve(network, _fleet(2), initial)
    else:
        result = solver.solve(network, _fleet(2), initial)
        assert result.paths == initial.paths
        assert result.stats.accepted == 0
        expected = 'timeout' if failure == 'timeout' else 'iteration_limit'
        assert result.status == expected
    expected_zones: Counter[tuple[str, int]] = Counter()
    expected_links: Counter[tuple[str, int]] = Counter()
    for path in initial.paths.values():
        for kind, name, turn in path.resources(network):
            table = expected_zones if kind == 'zone' else expected_links
            table[name, turn] += 1
    assert len(captured) == 1
    assert captured[0]._zone_use == expected_zones
    assert captured[0]._link_use == expected_links
    assert calls == 2


def test_zero_iteration_limit_preserves_the_seed() -> None:
    """Disabled repair has explicit termination metadata."""
    network = DroneNetwork.from_config(_config())
    initial = PrioritizedPlanner().solve(network, _fleet())
    result = LNSImprover(60, 0).solve(network, _fleet(), initial)
    assert result.paths == initial.paths
    assert result.status == 'iteration_limit'
    assert result.stats.iterations == result.stats.accepted == 0


@pytest.mark.parametrize('seconds', [-1.0, float('inf'), float('nan')])
def test_invalid_time_budget_is_rejected(seconds: float) -> None:
    """CLI-independent callers receive useful parameter errors.

    Args:
        seconds (float): Invalid duration passed to the solver constructor.
    """
    with pytest.raises(ValueError, match='time budget'):
        LNSImprover(seconds)


def test_negative_iteration_limit_is_rejected() -> None:
    """A typo must not silently disable improvement."""
    with pytest.raises(ValueError, match='iteration limit'):
        LNSImprover(max_iter=-1)


def test_incomplete_seed_is_rejected() -> None:
    """An incumbent for another fleet cannot silently drop deliveries."""
    network = DroneNetwork.from_config(_config())
    initial = PrioritizedPlanner().solve(network, _fleet(2))
    with pytest.raises(ValueError, match='requested fleet'):
        LNSImprover().solve(network, _fleet(), initial)
