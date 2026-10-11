"""Native flow relaxations providing rigorous makespan lower bounds.

For a resource r, let a(r) be its earliest possible use by a drone from
the start and b(r) its minimum remaining travel time to the goal. A run
finishing at M can use r at most cap(r) * max(0, M-a(r)-b(r)+1) times.
Every run induces a flow of N through these aggregate resource budgets.
An insufficient maximum flow therefore proves M impossible. These are
necessary conditions only: a sufficient aggregate flow is not a schedule.

Split vertices enforce hub capacity. Each undirected link has one shared
capacity arc, so opposing traversal directions cannot double its budget.
Restricted link use is the departure turn only; its extra travel turn is
included in b(r), consistent with releasing the link on arrival.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fly_in.models import ZoneType
from fly_in.realm.entities import Zone
from fly_in.realm.network import DroneNetwork
from fly_in.realm.routes import NativeDijkstra

if TYPE_CHECKING:
    from fly_in.solvers.base import SearchBudget


@dataclass(frozen=True)
class InstanceProfile:
    """Static instance features and a necessary completion-time bound."""

    fleet_size: int
    vertices: int
    edges: int
    density: float
    bottleneck_capacity: int
    restricted_ratio: float
    shortest_distance: int
    lower_bound: int


@dataclass
class _Arc:
    """Residual edge paired with its reverse arc."""

    destination: int
    reverse: int
    capacity: int


class _Flow:
    """Edmonds-Karp integer max flow, with no recursion or graph package."""

    def __init__(self, budget: SearchBudget | None = None) -> None:
        """Initialize residual adjacency and optional interruption checks."""
        self.budget = budget
        self.adjacency: list[list[_Arc]] = []

    def vertex(self) -> int:
        """Allocate a vertex and return its index."""
        self.adjacency.append([])
        return len(self.adjacency) - 1

    def connect(self, source: int, target: int, capacity: int) -> None:
        """Create a directed edge and its residual reverse."""
        forward = _Arc(target, len(self.adjacency[target]), capacity)
        backward = _Arc(source, len(self.adjacency[source]), 0)
        self.adjacency[source].append(forward)
        self.adjacency[target].append(backward)

    def maximum(self, source: int, target: int, limit: int) -> int:
        """Augment shortest residual paths until saturated or at limit."""
        total = 0
        while total < limit:
            if self.budget is not None:
                self.budget.check()
            previous: dict[int, tuple[int, int]] = {source: (source, -1)}
            queue = deque([source])
            while queue and target not in previous:
                if self.budget is not None:
                    self.budget.check()
                node = queue.popleft()
                for index, arc in enumerate(self.adjacency[node]):
                    if arc.capacity > 0 and arc.destination not in previous:
                        previous[arc.destination] = (node, index)
                        queue.append(arc.destination)
            if target not in previous:
                break
            amount = limit - total
            node = target
            while node != source:
                parent, index = previous[node]
                amount = min(amount, self.adjacency[parent][index].capacity)
                node = parent
            node = target
            while node != source:
                parent, index = previous[node]
                arc = self.adjacency[parent][index]
                arc.capacity -= amount
                self.adjacency[node][arc.reverse].capacity += amount
                node = parent
            total += amount
        return total


@dataclass(frozen=True)
class ResourceWindow:
    """Bound aggregate resource use for one fleet and candidate makespan.

    Independent earliest-use and remaining-distance minima relax correlations
    between routes. Sufficient aggregate capacity cannot prove feasibility.
    """

    fleet_size: int
    makespan: int | None = None

    def capacity(self, base: int, earliest: int, remaining: int) -> int:
        """Return static capacity or a fleet-capped aggregate use budget.

        For a candidate makespan, multiply capacity by the number of
        possible use turns that leave enough remaining travel time.
        This relaxed budget is necessary for feasibility, not a schedule.
        """
        if self.makespan is None:
            return base
        return min(
            self.fleet_size,
            base * max(0, self.makespan - earliest - remaining + 1),
        )


class NetworkProfiler:
    """Own native latency and capacity analysis for one network preparation.

    Topology and resource capacities must remain immutable during preparation.
    Each profile rebuilds its distances and residual flow without caching.
    Bind a new profiler when preparing a different network.
    Optional interruption checks share the caller's deadline and cancellation.
    """

    def __init__(
        self,
        network: DroneNetwork,
        budget: SearchBudget | None = None,
    ) -> None:
        """Bind network topology and an optional cooperative search budget."""
        self.network = network
        self.budget = budget
        self.routes = NativeDijkstra(network, self._entry_score, budget)

    def _check_budget(self) -> None:
        """Honor cancellation and deadlines when a budget is supplied."""
        if self.budget is not None:
            self.budget.check()

    def profile(self, fleet_size: int) -> InstanceProfile:
        """Profile a network and prove a lower bound without running a solver.

        Blocked zones are omitted from routing and flow. Counts and ratios
        describe the original map, including blocked and disconnected hubs.
        The flow bottleneck includes both regular hub and shared link limits;
        endpoints are unlimited. Unreachable goals and nonpositive fleets
        raise ValueError instead of reporting a misleading numeric bound.

        A single shortest route pipelined one departure per turn finishes by
        L+N-1 under arrival-turn connection release. Binary search up to that
        upper bound finds the first M satisfying the aggregate flow relaxation.
        Equality of a valid schedule's makespan and this bound proves optimal
        makespan, without the total path cost optimization guarantee.
        """
        self._check_budget()
        if fleet_size < 1:
            raise ValueError("fleet size must be a positive integer")
        from_start = self._distances(self.network.start.name)
        if self.network.end.name not in from_start:
            raise ValueError(
                f"end hub '{self.network.end.name}' is unreachable from "
                f"start hub '{self.network.start.name}'"
            )
        to_goal = self._distances(self.network.end.name, reverse=True)
        shortest = from_start[self.network.end.name]
        unlimited = 1 + sum(
            edge.max_link_capacity
            for edge in self.network.connections.values()
        )
        bottleneck = self._capacity_flow(from_start, to_goal, unlimited)
        lower, upper = shortest, shortest + fleet_size - 1
        while lower < upper:
            self._check_budget()
            middle = (lower + upper) // 2
            achievable = self._capacity_flow(
                from_start,
                to_goal,
                fleet_size,
                middle,
            )
            if achievable >= fleet_size:
                upper = middle
            else:
                lower = middle + 1

        vertices = len(self.network.zones)
        edges = len(self.network.connections)
        restricted = sum(
            zone.zone_type is ZoneType.RESTRICTED
            for zone in self.network.zones.values()
        )
        return InstanceProfile(
            fleet_size,
            vertices,
            edges,
            edges / vertices,
            bottleneck,
            restricted / vertices,
            shortest,
            lower,
        )

    def _distances(
        self,
        origin: str,
        reverse: bool = False,
    ) -> dict[str, int]:
        """Return physical durations from the shared route-score analysis."""
        return {
            name: score[0]
            for name, score in self.route_scores(origin, reverse).items()
        }

    def route_scores(
        self,
        origin: str,
        reverse: bool = False,
    ) -> dict[str, tuple[int, int]]:
        """Minimize duration, then entries into non-priority destinations.

        Forward costs use the destination's movement duration; reverse costs
        use the current zone's duration and priority flag. Blocked zones are
        excluded from expansion. Fresh score maps are returned for each call.
        Priority breaks equal-duration ties without changing physical bounds.
        """
        return self.routes.analyze(origin, reverse).scores

    @staticmethod
    def _entry_score(zone: Zone) -> tuple[int, int]:
        """Charge physical entry duration with priority as a secondary tie."""
        return (
            zone.movement_cost(),
            int(zone.zone_type is not ZoneType.PRIORITY),
        )

    def _capacity_flow(
        self,
        from_start: dict[str, int],
        to_goal: dict[str, int],
        limit: int,
        makespan: int | None = None,
    ) -> int:
        """Compute shared-resource flow or an aggregate time-window bound."""
        flow = _Flow(self.budget)
        vertices: dict[str, tuple[int, int]] = {}

        window = ResourceWindow(limit, makespan)

        for name in sorted(from_start):
            self._check_budget()
            zone = self.network.zones[name]
            incoming, outgoing = flow.vertex(), flow.vertex()
            vertices[name] = (incoming, outgoing)
            bound = limit
            if zone not in (self.network.start, self.network.end):
                if zone.max_drones is not None:
                    bound = window.capacity(
                        zone.max_drones,
                        from_start[name],
                        to_goal[name],
                    )
            flow.connect(incoming, outgoing, bound)

        for connection in self.network.connections.values():
            self._check_budget()
            first, second = connection.zone_a, connection.zone_b
            if first.name not in vertices or second.name not in vertices:
                continue
            incoming, outgoing = flow.vertex(), flow.vertex()
            # Independent direction minima intentionally relax correlations:
            # they may weaken a bound, but cannot rule out a legal schedule.
            earliest = min(from_start[first.name], from_start[second.name]) + 1
            remaining = min(
                first.movement_cost() - 1 + to_goal[first.name],
                second.movement_cost() - 1 + to_goal[second.name],
            )
            bound = window.capacity(
                connection.max_link_capacity,
                earliest,
                remaining,
            )
            flow.connect(incoming, outgoing, bound)
            for zone in (first, second):
                zone_in, zone_out = vertices[zone.name]
                flow.connect(zone_out, incoming, limit)
                flow.connect(outgoing, zone_in, limit)

        return flow.maximum(
            vertices[self.network.start.name][1],
            vertices[self.network.end.name][0],
            limit,
        )
