"""Rung 2: budgeted DFS over priority DAGs with a feasible incumbent.

The method follows consistent prioritized planning: each agent avoids the
aggregate reservations of every ancestor in the priority DAG. For a
capacity-k conflict, one pair from its k+1 agents is ordered per branch.
That may leave the conflict present, so further ordering branches remain
necessary. A successful partial ordering is feasible; this search does not
claim globally optimal makespan when its tree or budget is exhausted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import isfinite

from fly_in.realm.network import DroneNetwork
from fly_in.solvers.base import (
    Conflict, ReservationTable, ScheduledSolver, SearchBudget, SearchStats,
    SearchTimeout, Solution, SpaceTimeAStar, TimedPath, AgentOrder,
    ConflictDetector,
)
from fly_in.solvers.prioritized import PrioritizedPlanner, SeedPlanner


@dataclass(frozen=True)
class PriorityDAG:
    """Immutable higher-to-lower priority relations over a fixed fleet."""

    agents: tuple[str, ...]
    edges: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def _children(self) -> dict[str, list[str]]:
        """Index successor edges in stable numerical drone order."""
        children: dict[str, list[str]] = {agent: [] for agent in self.agents}
        for high, low in self.edges:
            children[high].append(low)
        for values in children.values():
            values.sort(key=AgentOrder.key)
        return children

    def has_cycle(self) -> bool:
        """Detect strongly connected components using iterative Tarjan DFS."""
        children = self._children()
        indices: dict[str, int] = {}
        low: dict[str, int] = {}
        active: set[str] = set()
        component_stack: list[str] = []
        for root in self.agents:
            if root in indices:
                continue
            indices[root] = low[root] = len(indices)
            component_stack.append(root)
            active.add(root)
            frames: list[tuple[str, Iterator[str]]] = [
                (root, iter(children[root])),
            ]
            while frames:
                agent, neighbors = frames[-1]
                child = next(neighbors, None)
                if child is not None:
                    if child == agent:
                        return True
                    if child not in indices:
                        indices[child] = low[child] = len(indices)
                        component_stack.append(child)
                        active.add(child)
                        frames.append((child, iter(children[child])))
                    elif child in active:
                        low[agent] = min(low[agent], indices[child])
                    continue
                frames.pop()
                if frames:
                    parent = frames[-1][0]
                    low[parent] = min(low[parent], low[agent])
                if low[agent] == indices[agent]:
                    size = 0
                    while True:
                        member = component_stack.pop()
                        active.remove(member)
                        size += 1
                        if member == agent:
                            break
                    if size > 1:
                        return True
        return False

    def add(self, high: str, low: str) -> PriorityDAG | None:
        """Return a new consistent ordering, or None when it makes a cycle."""
        if high not in self.agents or low not in self.agents:
            raise ValueError('priority relation refers to an unknown drone')
        candidate = PriorityDAG(self.agents, self.edges | {(high, low)})
        return None if candidate.has_cycle() else candidate

    def topological(self) -> tuple[str, ...]:
        """Order higher agents before lower ones, breaking ties numerically."""
        children = self._children()
        incoming = {agent: 0 for agent in self.agents}
        for _, low in self.edges:
            incoming[low] += 1
        queue = [(AgentOrder.key(agent), agent) for agent in self.agents
                 if incoming[agent] == 0]
        # Agents supplied directly by callers need not already be sorted.
        queue.sort()
        order = []
        while queue:
            _, agent = heappop(queue)
            order.append(agent)
            for child in children[agent]:
                incoming[child] -= 1
                if not incoming[child]:
                    heappush(queue, (AgentOrder.key(child), child))
        if len(order) != len(self.agents):
            raise ValueError('priority relations contain a cycle')
        return tuple(order)

    def ancestors(self) -> dict[str, set[str]]:
        """Find transitive higher-priority sets for reservations."""
        children = self._children()
        ancestors: dict[str, set[str]] = {
            agent: set() for agent in self.agents
        }
        for agent in self.topological():
            for child in children[agent]:
                ancestors[child].update(ancestors[agent])
                ancestors[child].add(agent)
        return ancestors


@dataclass
class _Node:
    """One consistent-priority search node and its current candidate paths."""

    priorities: PriorityDAG
    paths: dict[str, TimedPath]


class PBSSolver(ScheduledSolver):
    """Explore partial priorities while retaining a valid floor on cutoff."""

    def __init__(
        self, time_budget: float = 1.0, node_budget: int = 10000,
    ) -> None:
        """Configure cooperative seconds and high-level node expansions."""
        super().__init__()
        if not isfinite(time_budget) or time_budget < 0:
            raise ValueError('PBS time budget must be finite and nonnegative')
        if node_budget < 0:
            raise ValueError('PBS node budget must be nonnegative')
        self.time_budget = time_budget
        self.node_budget = node_budget

    @staticmethod
    def _pair(
        conflict: Conflict, ancestors: dict[str, set[str]],
    ) -> tuple[str, str] | None:
        """Choose an unordered pair from the minimal overcapacity set."""
        for index, first in enumerate(conflict.agents):
            for second in conflict.agents[index+1:]:
                if (first not in ancestors[second]
                        and second not in ancestors[first]):
                    return first, second
        return None

    @staticmethod
    def _replan(
        engine: SpaceTimeAStar, node: _Node, priorities: PriorityDAG,
        lower: str, horizon: int, budget: SearchBudget,
    ) -> dict[str, TimedPath] | None:
        """Repair the lower agent and descendants against ancestors."""
        ancestors = priorities.ancestors()
        paths = dict(node.paths)
        for agent in priorities.topological():
            budget.check()
            if agent != lower and lower not in ancestors[agent]:
                continue
            reservations = ReservationTable(engine.network)
            for higher in sorted(ancestors[agent], key=AgentOrder.key):
                reservations.reserve(paths[higher])
            path = engine.find_path(
                agent, reservations, horizon=horizon, budget=budget,
            )
            if path is None:
                return None
            paths[agent] = path
        return paths

    def solve(
        self, network: DroneNetwork, drone_ids: list[str],
        initial: Solution | None = None, budget: SearchBudget | None = None,
    ) -> Solution:
        """Run a genuine priority-DAG DFS; return the best feasible schedule.

        Finishing the initial floor uses a soft deadline but remains
        cancellable. Every subsequent A* and high-level branch observes the
        shared deadline. Only explicit cancellation escapes to the caller.
        """
        limits = (budget or SearchBudget()).child(self.time_budget)
        SearchBudget(cancelled=limits.cancelled).check()
        floor = SeedPlanner(PrioritizedPlanner()).obtain(
            network, drone_ids, initial, budget or SearchBudget(),
        )
        stats = SearchStats(expanded=0 if initial else floor.stats.expanded)
        best = Solution(dict(floor.paths), stats=stats)
        engine: SpaceTimeAStar | None = None
        try:
            limits.check()
            if self.node_budget == 0:
                best.status = 'node_limit'
                best.reason = 'PBS node budget exhausted'
                return best
            engine = SpaceTimeAStar(network, limits)
            agents = tuple(sorted(drone_ids, key=AgentOrder.key))
            paths: dict[str, TimedPath] = {}
            for agent in agents:
                path = engine.find_path(agent, horizon=best.makespan)
                if path is None:
                    best.reason = 'no independent path within incumbent span'
                    return best
                paths[agent] = path
            stack = [_Node(PriorityDAG(agents), paths)]
            seen: set[frozenset[tuple[str, str]]] = {frozenset()}
            while stack:
                limits.check()
                if stats.nodes >= self.node_budget:
                    best.status = 'node_limit'
                    best.reason = 'PBS node budget exhausted'
                    break
                node = stack.pop()
                stats.nodes += 1
                conflicts = ConflictDetector(network).find(node.paths)
                if not conflicts:
                    candidate = Solution(node.paths)
                    if candidate.objective < best.objective:
                        best.paths = candidate.paths
                        stats.accepted += 1
                    continue
                ancestors = node.priorities.ancestors()
                pair = self._pair(conflicts[0], ancestors)
                if pair is None:
                    # Aggregate ancestor avoidance should make this
                    # impossible; keep the valid incumbent if encountered.
                    continue
                children: list[_Node] = []
                for higher, lower in (pair, tuple(reversed(pair))):
                    limits.check()
                    priorities = node.priorities.add(higher, lower)
                    if priorities is None:
                        continue
                    closure = frozenset(
                        (high, low) for low, parents
                        in priorities.ancestors().items() for high in parents
                    )
                    if closure in seen:
                        continue
                    seen.add(closure)
                    repaired = self._replan(
                        engine, node, priorities, lower,
                        best.makespan, limits,
                    )
                    if repaired is not None:
                        children.append(_Node(priorities, repaired))
                # Preserve deterministic depth-first exploration. Crucially,
                # retain branches that still have the same capacity conflict.
                stack.extend(reversed(children))
        except SearchTimeout:
            best.status = 'timeout'
            best.reason = 'PBS time budget exhausted'
        finally:
            if engine is not None:
                stats.expanded += engine.expanded
        return best
