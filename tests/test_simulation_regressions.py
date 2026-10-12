"""Regression tests for simultaneous turns and deterministic planning."""

from collections import Counter
from dataclasses import dataclass
import random
from unittest.mock import patch

import pytest

from fly_in.parser import Parser
from fly_in.realm.entities import Drone
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solvers.greedy import GreedySolver
from fly_in.solvers.prioritized import PlanningCancelled, PriorityBasedSolver
from fly_in.solvers.protocol import Schedule


@dataclass
class Script:
    """Supply explicit rows including deliberate invalid movements."""

    rows: Schedule

    @property
    def has_pending_actions(self) -> bool:
        """Whether a scripted action row remains.

        Returns:
            bool: ``True`` while future rows remain, allowing planned waits.
        """
        return bool(self.rows)

    def plan(self, network: DroneNetwork, drones: list[Drone]) -> Schedule:
        """Consume and return the next scripted movement row.

        Args:
            network (DroneNetwork): Network being simulated; unused here.
            drones (list[Drone]): Current fleet; unused by this fixed script.

        Returns:
            Schedule: Next action row, or one empty row after exhaustion.
        """
        return [self.rows.pop(0)] if self.rows else [[]]


RESTRICTED = """nb_drones: 2
start_hub: s 0 0
end_hub: g 2 0
hub: r 1 0 [zone=restricted]
connection: s-r [max_link_capacity=2]
connection: r-g
"""


def test_restricted_arrival_uses_its_actual_turn() -> None:
    """A second departure can overlap the first drone's arrival."""
    sim = Simulation.from_config(
        Parser().parse_text(RESTRICTED), PriorityBasedSolver())
    assert sim.run().turns == 4
    assert sim.history[1] == ['D2-r-s', 'D1-r']
    assert sim.metrics.summary().total_path_cost == 6


def test_duplicate_actions_are_transactionally_rejected() -> None:
    """An invalid row preserves traffic, drones, metrics, and history."""
    script = Script([['D1-r-s', 'D1-r-s']])
    sim = Simulation.from_config(Parser().parse_text(RESTRICTED), script)
    sim.network.connections['r-s'].current_traffic = 1
    before = sim.metrics.summary()
    with pytest.raises(ValueError, match='duplicate'):
        sim.step()
    assert sim.turn == 0 and sim.history == []
    assert sim.metrics.summary() == before
    assert all(drone.current_zone is sim.network.start for drone in sim.drones)
    assert all(drone.in_transit is None for drone in sim.drones)
    assert sim.network.connections['r-s'].current_traffic == 1


def test_restricted_connection_releases_on_arrival_turn() -> None:
    """A new entry may share the turn when an earlier drone exits."""
    text = RESTRICTED.replace('max_link_capacity=2', 'max_link_capacity=1')
    sim = Simulation.from_config(Parser().parse_text(text),
                                 Script([['D1-r-s'], ['D2-r-s']]))
    sim.step()
    assert sim.step() == ['D2-r-s', 'D1-r']
    assert sim.turn == 2
    assert sim.drones[0].in_transit is None
    assert sim.drones[1].in_transit is not None
    assert sim.network.connections['r-s'].current_traffic == 1


def test_restricted_arrival_cannot_wait_for_destination_capacity() -> None:
    """A scheduled arrival into a full hub fails without delaying transit."""
    sim = Simulation.from_config(
        Parser().parse_text(RESTRICTED),
        Script([['D1-r-s'], ['D2-r-s'], []]),
    )
    sim.step()
    sim.step()
    assert sim.drones[0].current_zone.name == 'r'
    assert sim.drones[1].in_transit is not None
    before = sim.metrics.summary()
    history = [list(row) for row in sim.history]
    with pytest.raises(ValueError, match="zone 'r' capacity exceeded"):
        sim.step()
    assert sim.turn == 2
    assert sim.history == history
    assert sim.metrics.summary() == before
    assert sim.drones[1].in_transit is not None
    assert sim.drones[1].transit_remaining == 1


def test_callback_and_result_do_not_alias_history() -> None:
    """Consumer modifications cannot corrupt stored movement rows."""
    sim = Simulation.from_config(Parser().parse_text(RESTRICTED),
                                 PriorityBasedSolver())
    result = sim.run(lambda simulation, moves: moves.clear())
    assert all(sim.history)
    result.history[0].clear()
    assert sim.history[0]


def test_planned_wait_is_not_deadlock() -> None:
    """A finite planned idle turn is valid."""
    text = 'nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\nconnection: s-g'
    sim = Simulation.from_config(Parser().parse_text(text),
                                 Script([[], ['D1-g']]))
    assert sim.run().turns == 2


def test_priority_tie_prefers_priority_zone() -> None:
    """Equal arrival and physical cost prefer priority entries."""
    text = '''nb_drones: 1
start_hub: s 0 0
end_hub: g 2 0
hub: normal 1 -1
hub: preferred 1 1 [zone=priority]
connection: s-normal
connection: normal-g
connection: s-preferred
connection: preferred-g
'''
    sim = Simulation.from_config(Parser().parse_text(text),
                                 PriorityBasedSolver())
    assert sim.run().history == [['D1-preferred'], ['D1-g']]


def test_preparation_cancellation_and_unreachable_errors() -> None:
    """Cancellation and disconnected topology have distinct diagnostics."""
    solver = PriorityBasedSolver()
    sim = Simulation.from_config(Parser().parse_text(RESTRICTED), solver)
    with pytest.raises(PlanningCancelled):
        solver.prepare(sim.network, sim.drones, lambda: True)
    assert sim.turn == 0
    text = RESTRICTED.replace('connection: r-g', '')
    sim = Simulation.from_config(Parser().parse_text(text), solver)
    with pytest.raises(ValueError, match='no accessible route'):
        solver.prepare(sim.network, sim.drones)
    with pytest.raises(RuntimeError, match='horizon exhausted'):
        bounded = PriorityBasedSolver(horizon=1)
        sim = Simulation.from_config(Parser().parse_text(RESTRICTED), bounded)
        bounded.prepare(sim.network, sim.drones)


def test_blocked_endpoint_is_rejected() -> None:
    """A blocked start cannot be used as an exception to accessibility."""
    text = RESTRICTED.replace('start_hub: s 0 0',
                              'start_hub: s 0 0 [zone=blocked]')
    with pytest.raises(ValueError, match='cannot be blocked'):
        Simulation.from_config(
            Parser().parse_text(text), PriorityBasedSolver())


def test_opposite_directions_share_one_link_capacity() -> None:
    """Two opposing departures cannot each claim a capacity-one link."""
    text = '''nb_drones: 2
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0
hub: b 2 0
connection: s-a
connection: s-b
connection: a-b
connection: b-g
'''
    sim = Simulation.from_config(Parser().parse_text(text),
                                 Script([['D1-a', 'D2-b'], ['D1-b', 'D2-a']]))
    sim.step()
    before = sim.metrics.summary()
    with pytest.raises(ValueError, match='connection'):
        sim.step()
    assert sim.turn == 1
    assert [drone.current_zone.name for drone in sim.drones] == ['a', 'b']
    assert sim.metrics.summary() == before


def test_waiting_does_not_create_avoidable_detours() -> None:
    """A queued drone waits at source instead of looping at equal arrival."""
    text = '''nb_drones: 3
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0
hub: b 2 0
connection: s-a
connection: a-b
connection: b-s
connection: b-g
'''
    sim = Simulation.from_config(Parser().parse_text(text),
                                 PriorityBasedSolver())
    result = sim.run()
    assert result.turns == 4
    assert result.metrics.total_path_cost == 6
    assert all(len(drone.move_log) == 2 for drone in sim.drones)


def _generated_map(case: int) -> str:
    """Reproduce one of thirty generated maps without changing its seed.

    Args:
        case (int): Case number used with the test's fixed generator seed.

    Returns:
        str: Generated map text.
    """
    generator = random.Random(52)
    names = ['s', 'a', 'b', 'c', 'd', 'g']
    for _ in range(case + 1):
        lines = ['nb_drones: 6', 'start_hub: s 0 0', 'end_hub: g 5 0']
        for index, name in enumerate(names[1:-1], 1):
            kind = generator.choice(['normal', 'restricted', 'priority'])
            capacity = generator.randint(1, 2)
            lines.append(f'hub: {name} {index} 0 '
                         f'[zone={kind} max_drones={capacity}]')
        pairs = {(names[index], names[index + 1]) for index in range(5)}
        pairs.update((first, second)
                     for index, first in enumerate(names)
                     for second in names[index + 1:]
                     if generator.random() < 0.25)
        for first, second in sorted(pairs):
            capacity = generator.randint(1, 2)
            lines.append(f'connection: {first}-{second} '
                         f'[max_link_capacity={capacity}]')
    return '\n'.join(lines)


@pytest.mark.parametrize('case', range(30), ids=lambda case: f'map-{case}')
@pytest.mark.parametrize('solver_name', ['priority', 'greedy'])
def test_generated_networks_preserve_turn_invariants(
    solver_name: str, case: int,
) -> None:
    """Check invariants for each original topology and strategy case.

    Args:
        solver_name (str): Registered solver strategy under test.
        case (int): Generated map case number.
    """
    solver = (PriorityBasedSolver() if solver_name == 'priority'
              else GreedySolver())
    sim = Simulation.from_config(Parser().parse_text(_generated_map(case)),
                                 solver)
    for _turn in range(150):
        if all(drone.delivered for drone in sim.drones):
            break
        before = {drone.id: (drone.current_zone, drone.in_transit)
                  for drone in sim.drones}
        moves = sim.step()
        events = dict(move.split('-', 1) for move in moves)
        assert len(events) == len(moves)
        occupancy = Counter(drone.current_zone.name for drone in sim.drones
                            if not drone.delivered
                            and drone.in_transit is None)
        usage: Counter[str] = Counter()
        for drone in sim.drones:
            source, transit = before[drone.id]
            if transit is not None:
                assert drone.in_transit is None
                assert drone.current_zone is transit.other_end(source)
                assert events[drone.id] == drone.current_zone.name
                # Transit exit frees the link for a new entry this turn.
            elif drone.id in events:
                edge = (drone.in_transit
                        or sim.network.connection_between(
                            source, drone.current_zone))
                usage[edge.identifier()] += 1
            if drone.delivered:
                assert drone.current_zone is sim.network.end
        for name, count in occupancy.items():
            zone = sim.network.zones[name]
            if zone is not sim.network.start:
                assert zone.has_capacity(count)
        for name, edge in sim.network.connections.items():
            assert usage[name] <= edge.max_link_capacity
            assert edge.current_traffic == usage[name]
    else:
        pytest.fail('connected generated map failed to finish')


def test_reachability_precheck_observes_cancellation() -> None:
    """Cancel within topology traversal before building a timed schedule."""
    solver = PriorityBasedSolver()
    sim = Simulation.from_config(Parser().parse_text(RESTRICTED), solver)
    calls = 0

    def cancelled() -> bool:
        """Signal cancellation on the second topology-check poll.

        Returns:
            bool: Whether the configured cancellation point was reached.
        """
        nonlocal calls
        calls += 1
        return calls >= 2

    with patch.object(solver, '_build_schedule', side_effect=AssertionError):
        with pytest.raises(PlanningCancelled):
            solver.prepare(sim.network, sim.drones, cancelled)
    assert calls == 2


def test_prepared_schedule_is_specific_to_network_and_fleet() -> None:
    """A reused solver independently prepares an identical new fleet."""
    config = Parser().parse_text(RESTRICTED)
    solver = PriorityBasedSolver()
    first = Simulation.from_config(config, solver)
    first_result = first.run()
    second = Simulation.from_config(config, solver)
    assert second.run().history == first_result.history
    assert first.history == first_result.history


def test_endpoint_capacity_api_is_unlimited() -> None:
    """Endpoint exception is also expressed by the zone's own method."""
    from fly_in.parser import Parser
    from fly_in.realm.network import DroneNetwork

    config = Parser().parse_text(
        "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 2 0\n"
        "hub: h 1 0\nconnection: s-h\nconnection: h-g\n"
    )
    network = DroneNetwork.from_config(config)
    assert network.start.has_capacity(10_000)
    assert network.end.has_capacity(10_000)
    assert not network.start.has_capacity(-1)
    assert not network.zones['h'].has_capacity(2)
