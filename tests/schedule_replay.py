"""Independent output replay using map declarations, not scheduling helpers."""

from collections import Counter
from dataclasses import dataclass

from fly_in.models import MapConfig, ZoneType


@dataclass(frozen=True)
class ReplayResult:
    """Physical cost and deliveries reconstructed from emitted movements.

    Attributes:
        cost (int): Sum of movement durations, excluding waits.
        arrivals (dict[str, int]): Drone IDs mapped to their delivery turns,
            starting at turn 1.
    """

    cost: int
    arrivals: dict[str, int]


def replay(config: MapConfig, history: list[list[str]]) -> ReplayResult:
    """Validate emitted movements independently of the scheduling code.

    Read only map declarations and output rows. Restricted transit requires
    one link token followed by a hub token on the next turn. Arrivals release
    links for departures on that turn; opposing departures share link
    capacity, and hub occupancy is checked after simultaneous moves.

    Args:
        config (MapConfig): Validated map that defines the fleet and
            capacities.
        history (list[list[str]]): Movement tokens grouped by turn, starting
            at turn 1. Empty rows represent idle turns.

    Returns:
        ReplayResult: Total physical movement cost and each drone's arrival
        turn after all drones reach the goal.

    Raises:
        AssertionError: A movement, capacity, mandatory arrival, or completion
            condition is invalid, including rows after fleet delivery.
    """
    zones = {zone.name: zone
             for zone in (config.start, *config.zones, config.end)}
    capacities = {zone.name: zone.max_drones for zone in config.zones}
    links = {"-".join(sorted((edge.from_hub, edge.to_hub))): edge
             for edge in config.connections}
    positions = {f"D{index}": ("hub", config.start.name)
                 for index in range(1, config.nb_drones + 1)}
    arrivals: dict[str, int] = {}
    cost = 0
    for turn, row in enumerate(history, 1):
        assert len(arrivals) < config.nb_drones, "rows after completion"
        following = dict(positions)
        traffic: Counter[str] = Counter()
        events: dict[str, str] = {}
        for token in row:
            agent, separator, target = token.partition("-")
            assert separator and agent in positions, token
            assert agent not in events and agent not in arrivals, token
            events[agent] = target
            kind, source = positions[agent]
            if kind == "transit":
                assert target == source, (turn, agent, "forced arrival")
                following[agent] = ("hub", target)
            elif target in zones:
                destination = zones[target]
                assert destination.zone_type not in (
                    ZoneType.BLOCKED, ZoneType.RESTRICTED,
                ), (turn, token, "illegal one-turn destination")
                link = "-".join(sorted((source, target)))
                assert link in links, (turn, token, "nonadjacent move")
                traffic[link] += 1
                cost += 1
                following[agent] = ("hub", target)
            else:
                assert target in links, (turn, token, "unknown link")
                edge = links[target]
                assert source in (edge.from_hub, edge.to_hub), token
                destination_name = (edge.to_hub if source == edge.from_hub
                                    else edge.from_hub)
                assert zones[destination_name].zone_type == ZoneType.RESTRICTED
                traffic[target] += 1
                cost += 2
                following[agent] = ("transit", destination_name)
            if following[agent] == ("hub", config.end.name):
                arrivals[agent] = turn
        for agent, (kind, _) in positions.items():
            assert kind != "transit" or agent in events, (
                turn, agent, "missing mandatory restricted arrival",
            )
        occupancy = Counter(name for kind, name in following.values()
                            if kind == "hub")
        for name, count in occupancy.items():
            if name not in (config.start.name, config.end.name):
                assert count <= capacities[name], (turn, name, count)
        for name, count in traffic.items():
            assert count <= links[name].max_link_capacity, (turn, name, count)
        positions = following
    assert len(arrivals) == config.nb_drones, "not all drones delivered"
    assert max(arrivals.values()) == len(history)
    return ReplayResult(cost, arrivals)
