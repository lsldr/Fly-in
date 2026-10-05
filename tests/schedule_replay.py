"""Independent output replay using map declarations, not scheduling helpers."""

from collections import Counter
from dataclasses import dataclass

from fly_in.models import MapConfig, ZoneType


@dataclass(frozen=True)
class ReplayResult:
    """Physical cost and deliveries obtained solely from movement rows."""

    cost: int
    arrivals: dict[str, int]


def replay(config: MapConfig, history: list[list[str]]) -> ReplayResult:
    """Check every emitted movement, forced arrival, wait and capacity.

    Only input declarations and output rows are read. Neither Simulation
    state, TimedPath, reservations, ConflictDetector nor flow code is used.
    Restricted transit is one emitted link row followed by a forced hub row.
    Arrivals free links for new entries on that turn. Opposing departures
    share the same link count; hub occupancy is counted after all moves.
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
