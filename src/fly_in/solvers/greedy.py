"""A shortest-path, turn-by-turn baseline solver."""

from collections import Counter
from dataclasses import dataclass

from fly_in.models import ZoneType
from fly_in.realm.entities import Drone, Zone
from fly_in.realm.network import DroneNetwork
from fly_in.realm.routes import NativeDijkstra
from fly_in.solvers.protocol import Schedule


@dataclass(frozen=True)
class WeightedRoute:
    """One greedy route's discounted duration and selected first hop.

    Attributes:
        distance (float): Weighted path duration, measured in quarter-turns.
        first_hop (Zone | None): Next hub, or None when the goal is
            inaccessible from the source.
    """

    distance: float
    first_hop: Zone | None


class WeightedRoutePlanner:
    """Discounted shortest-route analyzer for one immutable topology.

    Priority entry duration is multiplied by 0.75. This heuristic preference
    differs from physical latency and never supplies an optimality bound.
    Equal distances keep the first predecessor; the queue breaks ties by name.
    Analyses are cached by source only within this bound network. Recreate the
    planner whenever topology or resource-zone types change.

    Attributes:
        network (DroneNetwork): Immutable topology analyzed by this planner.
        routes (NativeDijkstra): Weighted shortest-route engine.
    """

    def __init__(self, network: DroneNetwork) -> None:
        """Bind a network and begin with no cached source analyses.

        Args:
            network (DroneNetwork): Topology kept fixed for this planner.
        """
        self.network = network
        self.routes = NativeDijkstra(network, self._entry_score)
        self._analyses: dict[str, WeightedRoute] = {}

    def analyze(self, source: Zone) -> WeightedRoute:
        """Return weighted distance and first hop from one Dijkstra run.

        Args:
            source (Zone): Hub from which to route.

        Returns:
            WeightedRoute: Cached route analysis, including an infinite
                distance and missing hop when the goal is unreachable.
        """
        if source.name not in self._analyses:
            self._analyses[source.name] = self._search(source)
        return self._analyses[source.name]

    def next_zone(self, source: Zone) -> Zone:
        """Return the cached first hop or report an inaccessible goal.

        Args:
            source (Zone): Hub from which to route.

        Returns:
            Zone: First hub on the weighted shortest route.

        Raises:
            ValueError: If no accessible route reaches the goal.
        """
        route = self.analyze(source)
        if route.first_hop is None:
            raise ValueError(
                f"no accessible route from '{source.name}' to "
                f"'{self.network.end.name}'"
            )
        return route.first_hop

    def _search(self, source: Zone) -> WeightedRoute:
        """Use strict distance improvements and stable queue-name ties.

        Args:
            source (Zone): Hub from which to search.

        Returns:
            WeightedRoute: Weighted shortest distance and first hop, or an
                infinite-distance result when the goal is unreachable.
        """
        network = self.network
        analysis = self.routes.analyze(source.name, goal=network.end.name)
        previous = analysis.predecessors
        if source is network.end:
            return WeightedRoute(0.0, network.end)
        if network.end.name not in previous:
            return WeightedRoute(float("inf"), None)
        step = network.end.name
        while previous[step] != source.name:
            step = previous[step]
        return WeightedRoute(
            analysis.scores[network.end.name][0] / 4, network.zones[step],
        )

    @staticmethod
    def _entry_score(zone: Zone) -> tuple[int, int]:
        """Express the original greedy weighting exactly in quarter turns.

        Priority zones have one-turn entry duration, discounted to three
        quarters. Other destinations retain their full physical duration.

        Args:
            zone (Zone): Destination whose entry cost is calculated.

        Returns:
            tuple[int, int]: Weighted duration in quarter-turns and a zero
                secondary score.
        """
        return (3 if zone.zone_type is ZoneType.PRIORITY
                else 4 * zone.movement_cost()), 0


class GreedySolver:
    """Turn-by-turn planner using weighted routes and capacity checks.

    Route analyses and forced departures are retained across turns for the
    same network and fleet objects.
    """

    def __init__(self) -> None:
        """Start without a bound topology, fleet, or committed departures."""
        self._network: DroneNetwork | None = None
        self._route_planner: WeightedRoutePlanner | None = None
        self._routes: dict[str, Zone] = {}
        self._distances: dict[str, float] = {}
        self._committed: dict[str, Zone] = {}
        self._fleet: tuple[Drone, ...] = ()

    def plan(self, network: DroneNetwork, drones: list[Drone]) -> Schedule:
        """Select departures while guaranteeing next-turn forced arrivals.

        Args:
            network (DroneNetwork): Current immutable topology.
            drones (list[Drone]): Current fleet positions and transit states.

        Returns:
            Schedule: One row containing legal departures for this turn.

        Raises:
            ValueError: If no route exists or a committed departure is no
                longer legal.
        """
        if network is not self._network:
            self._network = network
            self._route_planner = WeightedRoutePlanner(network)
            self._routes.clear()
            self._distances.clear()
        if (len(drones) != len(self._fleet)
                or any(a is not b for a, b in zip(drones, self._fleet))):
            self._committed.clear()
            self._fleet = tuple(drones)
        by_id = {drone.id: drone for drone in drones}
        routes = {
            drone.id: self._route(network, drone.current_zone)
            for drone in drones
            if not drone.delivered and drone.in_transit is None
            and drone.current_zone is not network.end
        }
        for destination in routes.values():
            if destination.name not in self._distances:
                assert self._route_planner is not None
                self._distances[destination.name] = (
                    self._route_planner.analyze(destination).distance
                )
        ordered = sorted(
            routes.items(),
            key=lambda item: (self._distances[item[1].name], item[0]),
        )
        selected = dict(self._committed)
        future = self._lookahead(network, drones, selected)
        if future is None:
            raise ValueError("greedy committed departures are no longer legal")
        for drone_id, destination in ordered:
            if drone_id in selected:
                continue
            trial = {**selected, drone_id: destination}
            commitments = self._lookahead(network, drones, trial)
            if commitments is not None:
                selected = trial
                future = commitments
        self._committed = future
        actions = []
        for drone_id, destination in selected.items():
            target = destination.name
            if destination.movement_cost() == 2:
                target = network.connection_between(
                    by_id[drone_id].current_zone, destination
                ).identifier()
            actions.append(f"{drone_id}-{target}")
        return [actions]

    def _route(self, network: DroneNetwork, source: Zone) -> Zone:
        """Cache a next hop for current or projected hub occupants.

        Args:
            network (DroneNetwork): Topology containing the source.
            source (Zone): Hub whose next hop is needed.

        Returns:
            Zone: First hop toward the goal.

        Raises:
            ValueError: If the goal cannot be reached from the source.
        """
        if source.name not in self._routes:
            assert self._route_planner is not None
            self._routes[source.name] = self._route_planner.next_zone(source)
        return self._routes[source.name]

    def _lookahead(
        self, network: DroneNetwork, drones: list[Drone],
        departures: dict[str, Zone],
    ) -> dict[str, Zone] | None:
        """Project two turn boundaries and reserve safe one-turn evacuations.

        Newly restricted departures arrive at the second boundary. If a hub
        would overflow then, only guaranteed one-turn departures may free it.
        Every candidate rechecks the whole projection, so later choices cannot
        invalidate commitments accepted for earlier restricted departures.

        Args:
            network (DroneNetwork): Topology and capacity limits.
            drones (list[Drone]): Current fleet state.
            departures (dict[str, Zone]): Proposed immediate destinations.

        Returns:
            dict[str, Zone] | None: Forced one-turn evacuations that keep the
                projection legal, or None if no safe projection exists.
        """
        positions: dict[str, Zone] = {}
        incoming: Counter[str] = Counter()
        usage: Counter[str] = Counter()
        for drone in drones:
            if drone.delivered:
                continue
            destination = departures.get(drone.id)
            if drone.in_transit is not None:
                if destination is not None:
                    return None
                zone = drone.in_transit.other_end(drone.current_zone)
            elif destination is not None:
                link = network.connection_between(
                    drone.current_zone, destination)
                usage[link.identifier()] += 1
                if usage[link.identifier()] > link.max_link_capacity:
                    return None
                if destination.movement_cost() == 2:
                    incoming[destination.name] += 1
                    continue
                zone = destination
            else:
                zone = drone.current_zone
            if zone is not network.end:
                positions[drone.id] = zone
        occupancy = Counter(zone.name for zone in positions.values())
        if any(not network.zones[name].has_capacity(amount)
               for name, amount in occupancy.items()):
            return None
        occupancy.update(incoming)
        commitments: dict[str, Zone] = {}
        next_usage: Counter[str] = Counter()
        while any(not network.zones[name].has_capacity(amount)
                  for name, amount in occupancy.items()):
            changed = False
            for drone_id, source in positions.items():
                if (drone_id in commitments
                        or source.has_capacity(occupancy[source.name])):
                    continue
                target = self._route(network, source)
                if target.movement_cost() != 1:
                    continue
                link = network.connection_between(source, target)
                if (next_usage[link.identifier()] >= link.max_link_capacity
                        or not target.has_capacity(occupancy[target.name]+1)):
                    continue
                commitments[drone_id] = target
                next_usage[link.identifier()] += 1
                occupancy[source.name] -= 1
                occupancy[target.name] += 1
                changed = True
            if not changed:
                return None
        return commitments
