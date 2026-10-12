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
    """Fresh scores and strict-improvement predecessors for one search.

    Attributes:
        scores (dict[str, tuple[int, int]]): Best discovered score pair by
            zone name.
        predecessors (dict[str, str]): Previous zone on each improved route.
    """

    scores: dict[str, tuple[int, int]]
    predecessors: dict[str, str]


class NativeDijkstra:
    """Dijkstra analyzer with an explicit entry-score policy.

    Nonnegative score pairs are ordered lexicographically. Equal scores retain
    the first predecessor; zone names break queue ties. No search state or
    cache survives an analysis call, so separate callers never share mutations.

    Attributes:
        network (DroneNetwork): Topology analyzed by this instance.
        cost (Callable[[Zone], tuple[int, int]]): Per-entry score policy.
        budget (SearchBudget | None): Optional search interruption checks.
    """

    def __init__(
        self, network: DroneNetwork,
        cost: Callable[[Zone], tuple[int, int]],
        budget: SearchBudget | None = None,
    ) -> None:
        """Bind topology, entry scoring, and optional interruption checks.

        Args:
            network (DroneNetwork): Topology traversed by this analyzer.
            cost (Callable[[Zone], tuple[int, int]]): Nonnegative score pair
                charged according to traversal direction.
            budget (SearchBudget | None): Optional object checked before each
                queue removal.
        """
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

        Args:
            origin (str): Name of the initial zone.
            reverse (bool): Traverse links backward and charge the current
                zone when true.
            goal (str | None): Optional zone at which to stop once settled.

        Returns:
            RouteAnalysis: Scores and predecessor links discovered by search.
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
