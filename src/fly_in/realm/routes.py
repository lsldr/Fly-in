"""Native weighted route analysis shared by physical and heuristic planners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import TYPE_CHECKING

from fly_in.models import ZoneType
from fly_in.realm.entities import Zone
from fly_in.realm.network import DroneNetwork

if TYPE_CHECKING:
    from fly_in.solvers.base import SearchBudget


@dataclass(frozen=True)
class RouteAnalysis:
    """Fresh scores and strict-improvement predecessors for one search."""

    scores: dict[str, tuple[int, int]]
    predecessors: dict[str, str]


class NativeDijkstra:
    """Analyze one immutable topology using an explicit entry-score policy.

    Nonnegative score pairs are ordered lexicographically. Equal scores retain
    the first predecessor; zone names break queue ties. No search state or
    cache survives an analysis call, so separate callers never share mutations.
    """

    def __init__(
        self, network: DroneNetwork,
        cost: Callable[[Zone], tuple[int, int]],
        budget: SearchBudget | None = None,
    ) -> None:
        """Bind topology, entry scoring, and optional interruption checks."""
        self.network = network
        self.cost = cost
        self.budget = budget

    def analyze(
        self, origin: str, reverse: bool = False, goal: str | None = None,
    ) -> RouteAnalysis:
        """Find route scores, optionally stopping when the goal is settled.

        Forward traversal charges the neighbor; reverse traversal charges the
        current zone. Blocked zones are never expanded or entered. A result
        limited to a goal may include unsettled scores outside its route.
        """
        scores = {origin: (0, 0)}
        predecessors: dict[str, str] = {}
        queue = [(0, 0, origin)]
        while queue:
            if self.budget is not None:
                self.budget.check()
            primary, secondary, name = heappop(queue)
            if (primary, secondary) != scores[name]:
                continue
            zone = self.network.zones[name]
            if zone.zone_type is ZoneType.BLOCKED:
                continue
            if name == goal:
                break
            for connection in self.network.adjacency[name]:
                neighbor = connection.other_end(zone)
                if neighbor.zone_type is ZoneType.BLOCKED:
                    continue
                entry = zone if reverse else neighbor
                first, second = self.cost(entry)
                candidate = primary + first, secondary + second
                if neighbor.name not in scores or (
                    candidate < scores[neighbor.name]
                ):
                    scores[neighbor.name] = candidate
                    predecessors[neighbor.name] = name
                    heappush(queue, (*candidate, neighbor.name))
        return RouteAnalysis(scores, predecessors)
