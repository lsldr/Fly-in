"""ICBS mechanisms and independent exhaustive small-instance optimality."""

import pytest

from joint_oracle import exact_makespan

from fly_in.models import ZoneType
from fly_in.realm.entities import Connection, Zone
from fly_in.realm.network import DroneNetwork
from fly_in.solvers.base import (
    PlanningCancelled, SearchBudget, Solution, SpaceTimeAStar, TimedMove,
    TimedPath, ConflictDetector,
)
from fly_in.solvers.icbs import ICBSSolver, _Node, ConflictAssessment


def _network(
    branches: int = 1, capacity: int = 1, link_capacity: int = 1,
    restricted: bool = False,
) -> DroneNetwork:
    """Construct short parallel routes without using solver primitives.

    Args:
        branches (int): Number of parallel intermediate routes in the generated
            network.
        capacity (int): Capacity assigned to each intermediate hub.
        link_capacity (int): Per-turn capacity assigned to each generated
            connection.
        restricted (bool): Whether intermediate hubs have two-turn entry
            duration.

    Returns:
        DroneNetwork: Topology built from the generated hubs and connections.
    """
    start = Zone('s', 0, 0, max_drones=None)
    end = Zone('g', 2, 0, max_drones=None)
    zones = {'s': start, 'g': end}
    adjacency: dict[str, list[Connection]] = {'s': [], 'g': []}
    for index in range(branches):
        name = chr(ord('a') + index)
        kind = ZoneType.RESTRICTED if restricted else ZoneType.NORMAL
        hub = Zone(name, 1, index, kind, capacity)
        zones[name] = hub
        adjacency[name] = []
        for source, target in ((start, hub), (hub, end)):
            edge = Connection(source, target, link_capacity)
            adjacency[source.name].append(edge)
            adjacency[target.name].append(edge)
    return DroneNetwork(zones, adjacency, start, end)


def _serial_seed(network: DroneNetwork, drones: int) -> Solution:
    """Build a deliberately loose feasible incumbent to exercise CT search.

    Args:
        network (DroneNetwork): Topology used to build the independent paths.
        drones (int): Number of paths to separate with start-hub waits.

    Returns:
        Solution: Feasible paths separated by waits to provide a loose floor.
    """
    engine = SpaceTimeAStar(network)
    paths = {}
    offset = 0
    for index in range(drones):
        agent = f'D{index+1}'
        path = engine.find_path(agent)
        assert path is not None
        waits = tuple((network.start.name, turn) for turn in range(offset))
        states = waits + tuple((name, time+offset)
                               for name, time in path.states)
        moves = tuple(TimedMove(move.source, move.destination, move.link,
                                move.departure+offset, move.arrival+offset)
                      for move in path.moves)
        paths[agent] = TimedPath(agent, states, moves)
        offset += path.arrival
    return Solution(paths)


@pytest.mark.parametrize('branches,capacity,link_capacity,restricted,drones', [
    (1, 1, 1, False, 1), (1, 1, 1, False, 3),
    (1, 2, 3, False, 3), (1, 2, 2, True, 3),
    (1, 1, 1, True, 3), (2, 1, 1, False, 3),
    (2, 1, 1, True, 3), (2, 2, 2, False, 4),
])
def test_icbs_matches_exhaustive_joint_oracle(
    branches: int, capacity: int, link_capacity: int, restricted: bool,
    drones: int,
) -> None:
    """Prove small makespans including capacity two and released transit.

    Args:
        branches (int): Number of parallel intermediate routes in the generated
            network.
        capacity (int): Capacity shared by generated intermediate hubs and
            connections.
        link_capacity (int): Per-turn capacity assigned to each generated
            connection.
        restricted (bool): Whether intermediate hubs have two-turn entry
            duration.
        drones (int): Fleet size used to build and solve the generated
            topology.
    """
    network = _network(branches, capacity, link_capacity, restricted)
    result = ICBSSolver(time_budget=10).solve(
        network, [f'D{i+1}' for i in range(drones)],
        initial=_serial_seed(network, drones),
    )
    assert result.optimal, result.reason
    assert not ConflictDetector(network).find(result.paths)
    assert result.makespan == exact_makespan(network, drones)


def test_capacity_two_conflict_has_three_children() -> None:
    """A pairwise split must not discard two legal simultaneous occupants."""
    network = _network(capacity=2, link_capacity=3)
    engine = SpaceTimeAStar(network)
    paths = {}
    for agent in ('D1', 'D2', 'D3'):
        path = engine.find_path(agent)
        assert path is not None
        paths[agent] = path
    node = _Node(frozenset(), paths)
    conflicts = ConflictDetector(network).find(paths)
    split = ICBSSolver()._select_split(
        node, conflicts, engine, 4, SearchBudget(),
    )
    assert split.classification == 'cardinal'
    assert len(split.children) == 3
    assert {next(iter(child.constraints)).agent for child in split.children}
    assert all(len(child.constraints) == 1 for child in split.children)


@pytest.mark.parametrize('bounds,expected', [
    ([6, 6], 'cardinal'), ([None, 6, None], 'cardinal'),
    ([5, 6], 'semi_cardinal'), ([5, 5, None], 'semi_cardinal'),
    ([5, 5], 'non_cardinal'),
])
def test_classification_compares_makespan_bounds(
    bounds: list[int | None], expected: str,
) -> None:
    """Unchanged global makespan is not cardinal just because a path grew.

    Args:
        bounds (list[int | None]): Child makespan bounds used to classify a
            conflict.
        expected (str): Conflict class expected for the supplied child bounds.
    """
    assert ConflictAssessment(5, bounds).classification() == expected


def test_bypass_replaces_paths_without_adding_constraints() -> None:
    """Equal-arrival alternate route removes conflicts before branching."""
    network = _network(branches=2)
    seed = _serial_seed(network, 2)
    result = ICBSSolver().solve(network, ['D1', 'D2'], initial=seed)
    assert result.optimal
    assert result.makespan == 2
    assert result.stats.bypasses > 0
    assert result.stats.non_cardinal > 0
    assert seed.makespan == 4


def test_bypass_reduces_excess_before_a_conflict_disappears() -> None:
    """A partial reroute from five users to four helps a capacity-three hub."""
    network = _network(branches=2, capacity=3, link_capacity=3)
    engine = SpaceTimeAStar(network)
    paths = {}
    for index in range(5):
        agent = f'D{index+1}'
        path = engine.find_path(agent)
        assert path is not None
        paths[agent] = path
    node = _Node(frozenset(), paths)
    conflicts = ConflictDetector(network).find(paths)
    assert all(item.excess == 2 for item in conflicts)
    assert all(len(item.agents) == 4 for item in conflicts)
    split = ICBSSolver()._select_split(
        node, conflicts, engine, 4, SearchBudget(),
    )
    bypass = ICBSSolver._bypass(
        network, node, conflicts, split, SearchBudget(),
    )
    assert bypass is not None
    remaining = ConflictDetector(network).find(bypass.paths)
    assert len(remaining) == len(conflicts)
    assert all(item.excess == 1 for item in remaining)
    assert bypass.constraints == node.constraints
    for agent, path in bypass.paths.items():
        assert (path.arrival, path.cost) == (
            paths[agent].arrival, paths[agent].cost,
        )


def test_cardinal_split_precedes_earlier_noncardinal_conflict() -> None:
    """A mandatory shared final link wins over an avoidable earlier link."""
    network = _network(branches=2)
    old_end = network.end
    old_end.max_drones = 2
    final = Zone('z', 3, 0, max_drones=None)
    link = Connection(old_end, final)
    network.zones['z'] = final
    network.adjacency['g'].append(link)
    network.adjacency['z'] = [link]
    network = DroneNetwork(network.zones, network.adjacency,
                           network.start, final)
    engine = SpaceTimeAStar(network)
    paths = {}
    for agent in ('D1', 'D2'):
        path = engine.find_path(agent)
        assert path is not None
        paths[agent] = path
    conflicts = ConflictDetector(network).find(paths)
    split = ICBSSolver()._select_split(
        _Node(frozenset(), paths), conflicts, engine, 6, SearchBudget(),
    )
    assert split.classification == 'cardinal'
    assert split.conflict.resource == 'g-z'


def test_bounded_search_retains_feasible_uncertified_incumbent() -> None:
    """Node and time budgets preserve the supplied floor with honest status."""
    network = _network()
    seed = _serial_seed(network, 3)
    for solver, status in ((ICBSSolver(time_budget=0), 'timeout'),
                           (ICBSSolver(node_budget=0), 'node_limit')):
        result = solver.solve(network, ['D1', 'D2', 'D3'], initial=seed)
        assert result.objective == seed.objective
        assert not result.optimal
        assert result.status == status
        assert not ConflictDetector(network).find(result.paths)


def test_cancellation_is_not_swallowed_as_a_timeout() -> None:
    """Explicit GUI cancellation propagates through preparation."""
    with pytest.raises(PlanningCancelled):
        ICBSSolver().solve(_network(), ['D1'],
                           budget=SearchBudget(cancelled=lambda: True))


@pytest.mark.parametrize('seconds', [-1, float('inf'), float('nan')])
def test_invalid_time_budget_is_rejected(seconds: float) -> None:
    """Invalid deadlines must not produce unlimited searches accidentally.

    Args:
        seconds (float): Invalid duration passed to the solver constructor.
    """
    with pytest.raises(ValueError):
        ICBSSolver(time_budget=seconds)


def test_negative_node_budget_is_rejected() -> None:
    """Validate the high-level expansion budget at construction."""
    with pytest.raises(ValueError):
        ICBSSolver(node_budget=-1)
