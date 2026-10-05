"""Exhaustive joint-state oracle independent of solver and bound algorithms."""

from collections import Counter, deque
from itertools import product

from fly_in.models import ZoneType
from fly_in.realm.network import DroneNetwork


Position = tuple[str, str]
JointState = tuple[Position, ...]


def _options(
    network: DroneNetwork, position: Position,
) -> list[tuple[Position, str | None]]:
    """Enumerate waits/departures, or force the next restricted arrival."""
    kind, name = position
    if kind == "transit":
        return [(("hub", name), None)]
    choices: list[tuple[Position, str | None]] = [(position, None)]
    if name == network.end.name:
        return choices
    zone = network.zones[name]
    for edge in network.adjacency[name]:
        target = edge.other_end(zone)
        if target.zone_type is ZoneType.BLOCKED:
            continue
        target_kind = ("transit" if target.zone_type is ZoneType.RESTRICTED
                       else "hub")
        choices.append(((target_kind, target.name), edge.identifier()))
    return choices


def exact_makespan(network: DroneNetwork, drones: int) -> int:
    """Find the minimum turns by BFS of all legal simultaneous decisions.

    This uses no solver paths, constraints, reservation tables, or flow
    bounds. A transit state must arrive next turn; only new departures
    count toward link capacity under arrival-turn release. Delivered and
    start occupancy are unlimited. Sorting collapses drone permutations.
    Exhausting reachable states establishes that delivery is impossible.
    Use only on small instances: joint state enumeration is exponential.
    """
    initial = tuple(("hub", network.start.name) for _ in range(drones))
    queue: deque[tuple[JointState, int]] = deque([(initial, 0)])
    visited = {initial}
    while queue:
        state, turn = queue.popleft()
        if all(position == ("hub", network.end.name) for position in state):
            return turn
        for choices in product(*[_options(network, pos) for pos in state]):
            following = tuple(sorted(position for position, _ in choices))
            occupancy = Counter(name for kind, name in following
                                if kind == "hub")
            traffic = Counter(link for _, link in choices if link is not None)
            if any(not network.zones[name].has_capacity(amount)
                   for name, amount in occupancy.items()):
                continue
            if any(amount > network.connections[name].max_link_capacity
                   for name, amount in traffic.items()):
                continue
            if following not in visited:
                visited.add(following)
                queue.append((following, turn + 1))
    raise ValueError("joint enumeration found no schedule")
