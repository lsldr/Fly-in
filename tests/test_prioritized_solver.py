"""Integration coverage for the prioritized space-time solver."""

from heapq import heappop, heappush
from pathlib import Path

from fly_in.models import ZoneType
from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solvers.prioritized import PriorityBasedSolver

CHALLENGER_MAP = (
    Path(__file__).resolve().parents[1]
    / "maps"
    / "challenger"
    / "01_the_impossible_dream.txt"
)


def _shortest_duration(network: DroneNetwork) -> int:
    """Return the shortest legal duration from the start to the end hub.

    Args:
        network (DroneNetwork): Topology used for the independent route, seed,
            or solver operation.

    Returns:
        int: Minimum physical travel duration from start to goal.
    """
    distances = {network.start.name: 0}
    queue: list[tuple[int, str]] = [(0, network.start.name)]
    while queue:
        duration, zone_name = heappop(queue)
        if duration != distances[zone_name]:
            continue
        if zone_name == network.end.name:
            return duration
        zone = network.zones[zone_name]
        for connection in network.adjacency[zone_name]:
            destination = connection.other_end(zone)
            if destination.zone_type is ZoneType.BLOCKED:
                continue
            candidate = duration + destination.movement_cost()
            if candidate < distances.get(destination.name, float("inf")):
                distances[destination.name] = candidate
                heappush(queue, (candidate, destination.name))
    raise AssertionError("challenger map has no route to its goal")


def test_priority_solver_achieves_the_challenger_optimum() -> None:
    """Find and validate the capacity-bound optimum through Simulation."""
    config = Parser().parse(CHALLENGER_MAP)
    network = DroneNetwork.from_config(config)

    assert _shortest_duration(network) == 19
    goal_links = network.adjacency[network.end.name]
    assert len(goal_links) == 1
    assert goal_links[0].max_link_capacity == 1
    lower_bound = 19 + config.nb_drones - 1
    assert lower_bound == 43

    simulation = Simulation.from_config(config, PriorityBasedSolver())
    result = simulation.run()

    assert result.turns == lower_bound
    assert all(drone.delivered for drone in simulation.drones)
    delivery_turns = [
        turn
        for turn, moves in enumerate(result.history, start=1)
        if any(move.endswith("-impossible_goal") for move in moves)
    ]
    assert delivery_turns == list(range(19, 44))
