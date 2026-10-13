"""Capacity-aware CBS with makespan conflict prioritization and bypasses.

The low level finds each constrained agent's earliest arrival. A conflict-tree
node's maximum arrival is consequently a makespan lower bound. Capacity c
requires c+1 children, each excluding a different member of a violating set.
The optional physical-cost tie-break does not prove secondary optimality.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from typing import Literal

from fly_in.realm.network import DroneNetwork
from fly_in.solvers.base import (
    Conflict, Constraint, ScheduledSolver, SearchBudget, SearchStats,
    SearchTimeout, Solution, SpaceTimeAStar, TimedPath, ConflictDetector,
    SolutionValidator,
)
from fly_in.solvers.bounds import NetworkProfiler
from fly_in.solvers.prioritized import PrioritizedPlanner, SeedPlanner

ConflictClass = Literal['cardinal', 'semi_cardinal', 'non_cardinal']


class ConflictAssessment:
    """Conflict classification from a parent's bound and complete split.

    Attributes:
        parent_bound (int): Parent's maximum individual earliest arrival.
        child_bounds (tuple[int | None, ...]): Feasible child bounds, with
            None marking a child whose constrained path is infeasible.
    """

    def __init__(
        self, parent_bound: int, child_bounds: Iterable[int | None],
    ) -> None:
        """Own a repeatable snapshot of all tested child bounds.

        Args:
            parent_bound (int): Parent CT node's makespan lower bound.
            child_bounds (Iterable[int | None]): Bound of each child, or None
                when that branch has no feasible low-level path.
        """
        self.parent_bound = parent_bound
        self.child_bounds = tuple(child_bounds)

    def classification(self) -> ConflictClass:
        """Treat infeasible children as infinite bound increases.

        Returns:
            ConflictClass: Cardinal, semi-cardinal, or non-cardinal class.

        Raises:
            ValueError: If no child bound was supplied.
        """
        increases = [bound is None or bound > self.parent_bound
                     for bound in self.child_bounds]
        if not increases:
            raise ValueError('a conflict must have at least one child')
        if all(increases):
            return 'cardinal'
        if any(increases):
            return 'semi_cardinal'
        return 'non_cardinal'


@dataclass(frozen=True)
class _Node:
    """Constraint set and individually earliest consistent candidate paths.

    Attributes:
        constraints (frozenset[Constraint]): Restrictions defining this node.
        paths (dict[str, TimedPath]): Individually earliest consistent paths.
    """

    constraints: frozenset[Constraint]
    paths: dict[str, TimedPath]

    @property
    def bound(self) -> int:
        """Return the admissible lower bound on this subtree's makespan.

        Returns:
            int: Latest arrival among the node's individually earliest paths.
        """
        return max((path.arrival for path in self.paths.values()), default=0)

    @property
    def cost(self) -> int:
        """Return physical cost as a deterministic secondary queue key.

        Returns:
            int: Sum of traversal durations across the node's paths.
        """
        return sum(path.cost for path in self.paths.values())


@dataclass(frozen=True)
class _Split:
    """One tested conflict and all feasible children of its complete split.

    Attributes:
        conflict (Conflict): Violation branched on.
        classification (ConflictClass): Bound-based conflict class.
        children (tuple[_Node, ...]): Feasible branches from excluding each
            involved agent's resource use.
    """

    conflict: Conflict
    classification: ConflictClass
    children: tuple[_Node, ...]


class ICBSSolver(ScheduledSolver):
    """Makespan-ordered constraint-tree search with a feasible floor.

    Conflict prioritization and helpful bypasses follow ICBS. Meta-agent
    merging is not enabled. Returned optimality concerns makespan alone;
    timeout and node-limit exits retain a feasible, uncertified incumbent.

    Attributes:
        time_budget (float | None): Optional improvement budget in seconds.
        node_budget (int | None): Optional CT expansion limit.
        proof_required (bool): Whether interruption must raise instead of
            returning an unproved floor.
    """

    def __init__(
        self, time_budget: float | None = 1.0,
        node_budget: int | None = 10000,
    ) -> None:
        """Set a cooperative deadline and CT expansion limit.

        Args:
            time_budget (float | None): Nonnegative seconds, or None for no
                solver-local time limit.
            node_budget (int | None): Nonnegative CT node count, or None for
                no solver-local node limit.

        Raises:
            ValueError: If either supplied limit is negative or non-finite.
        """
        super().__init__()
        if time_budget is not None and (
            not isfinite(time_budget) or time_budget < 0
        ):
            raise ValueError('time budget must be finite and non-negative')
        if node_budget is not None and node_budget < 0:
            raise ValueError('node budget must be non-negative')
        self.time_budget = time_budget
        self.node_budget = node_budget
        self.proof_required = False

    def solve(
        self, network: DroneNetwork, drone_ids: list[str],
        initial: Solution | None = None, budget: SearchBudget | None = None,
    ) -> Solution:
        """Return the best feasible schedule with honest proof status.

        Args:
            network (DroneNetwork): Topology to plan.
            drone_ids (list[str]): Fleet identifiers to schedule.
            initial (Solution | None): Optional feasible incumbent.
            budget (SearchBudget | None): Parent deadline and cancellation.

        Returns:
            Solution: Best feasible result; status records whether search
                proved its makespan optimal.

        Raises:
            ValueError: If the fleet or incumbent is invalid.
            PlanningCancelled: If cancellation is requested.
            SearchTimeout: If no feasible floor exists when a required proof
                budget expires.
        """
        parent = budget or SearchBudget()
        limits = parent.child(self.time_budget)
        incumbent: Solution | None = None
        engine: SpaceTimeAStar | None = None
        try:
            if self.proof_required and initial is None:
                limits.check()
            floor = SeedPlanner(PrioritizedPlanner()).obtain(
                network, drone_ids, initial,
                limits if self.proof_required else parent,
            )
            stats = SearchStats(
                expanded=0 if initial else floor.stats.expanded,
            )
            incumbent = Solution(dict(floor.paths), stats=stats)
            # Heuristic floor construction may overrun its local improvement
            # timer. The caller's earlier deadline still governs certification.
            proof_budget = limits if self.proof_required else parent
            proof_budget.check()
            incumbent.lower_bound = NetworkProfiler(
                network, proof_budget,
            ).profile(len(drone_ids)).lower_bound
            if incumbent.makespan == incumbent.lower_bound:
                proof_budget.check()
                return self._proved(incumbent, 'certified flow lower bound')
            limits.check()
            engine = SpaceTimeAStar(network, limits)
            root_paths: dict[str, TimedPath] = {}
            for agent in drone_ids:
                path = engine.find_path(agent, horizon=incumbent.makespan)
                if path is None:
                    raise ValueError('incumbent does not fit the network')
                root_paths[agent] = path
            incumbent = self._search(
                network, engine, _Node(frozenset(), root_paths),
                incumbent, limits,
            )
            SolutionValidator(network).validate(incumbent, drone_ids, limits)
        except SearchTimeout:
            if incumbent is None:
                raise
            incumbent.optimal = False
            incumbent.status = 'timeout'
            incumbent.reason = 'ICBS planning deadline exhausted'
        finally:
            if engine is not None and incumbent is not None:
                incumbent.stats.expanded += engine.expanded
        return incumbent

    def _search(
        self, network: DroneNetwork, engine: SpaceTimeAStar, root: _Node,
        incumbent: Solution, budget: SearchBudget,
    ) -> Solution:
        """Maintain complete CT branching under an incumbent makespan bound.

        Args:
            network (DroneNetwork): Topology used for conflict detection.
            engine (SpaceTimeAStar): Low-level constrained path search.
            root (_Node): Initial conflict-tree node.
            incumbent (Solution): Feasible upper bound to improve or prove.
            budget (SearchBudget): Cooperative limits for the search.

        Returns:
            Solution: Updated incumbent with proof or cutoff status.

        Raises:
            PlanningCancelled: If cancellation is requested.
            SearchTimeout: If the budget deadline elapses.
        """
        serial = count()
        pending = [(root.bound, root.cost, next(serial), root)]
        visited = {root.constraints}
        stats = incumbent.stats
        while pending:
            budget.check()
            if pending[0][0] >= incumbent.makespan:
                return self._proved(incumbent, 'minimum OPEN bound')
            budget.check()
            if (self.node_budget is not None
                    and stats.nodes >= self.node_budget):
                incumbent.status = 'node_limit'
                incumbent.reason = 'ICBS constraint-tree node limit reached'
                return incumbent
            _, _, _, node = heappop(pending)
            stats.nodes += 1
            conflicts = ConflictDetector(network).find(node.paths)
            if not conflicts:
                candidate = Solution(dict(node.paths), stats=stats)
                if candidate.objective < incumbent.objective:
                    incumbent.paths = candidate.paths
                return self._proved(incumbent, 'minimum conflict-free CT node')
            split = self._select_split(
                node, conflicts, engine, incumbent.makespan, budget,
            )
            setattr(stats, split.classification,
                    getattr(stats, split.classification) + 1)
            bypass = self._bypass(network, node, conflicts, split, budget)
            if bypass is not None:
                stats.bypasses += 1
                heappush(pending, (bypass.bound, bypass.cost,
                                   next(serial), bypass))
                continue
            for child in split.children:
                budget.check()
                if child.constraints in visited:
                    continue
                visited.add(child.constraints)
                if child.bound >= incumbent.makespan:
                    # Equal makespan candidates can still improve physical
                    # cost, though this is not a secondary optimality proof.
                    if (not ConflictDetector(network).find(child.paths)
                            and child.cost < incumbent.cost):
                        incumbent.paths = dict(child.paths)
                    continue
                heappush(pending, (child.bound, child.cost,
                                   next(serial), child))
        budget.check()
        return self._proved(incumbent, 'exhausted all improving CT branches')

    @staticmethod
    def _proved(incumbent: Solution, reason: str) -> Solution:
        """Attach an explicit makespan-only optimality certificate.

        Args:
            incumbent (Solution): Feasible result to mark as proved.
            reason (str): Search condition establishing the certificate.

        Returns:
            Solution: The same solution, marked optimal for makespan.
        """
        incumbent.optimal = True
        incumbent.status = 'optimal'
        incumbent.reason = f'makespan proved by {reason}'
        return incumbent

    def _select_split(
        self, node: _Node, conflicts: list[Conflict],
        engine: SpaceTimeAStar, horizon: int, budget: SearchBudget,
    ) -> _Split:
        """Probe complete splits and prefer cardinal, then semi-cardinal.

        Args:
            node (_Node): Conflict-tree node being branched.
            conflicts (list[Conflict]): Ordered violations in the node.
            engine (SpaceTimeAStar): Constrained path search engine.
            horizon (int): Current incumbent makespan limit.
            budget (SearchBudget): Cooperative interruption controls.

        Returns:
            _Split: Preferred fully probed split and its feasible children.

        Raises:
            ValueError: If called with no conflicts.
            PlanningCancelled: If cancellation is requested.
            SearchTimeout: If the budget deadline elapses.
        """
        preferred: _Split | None = None
        rank = {'cardinal': 0, 'semi_cardinal': 1, 'non_cardinal': 2}
        for conflict in conflicts:
            budget.check()
            children: list[_Node] = []
            bounds: list[int | None] = []
            for agent in conflict.agents:
                budget.check()
                constraints = node.constraints | {conflict.constraint(agent)}
                path = engine.find_path(
                    agent, constraints=constraints, horizon=horizon,
                    budget=budget,
                )
                if path is None:
                    bounds.append(None)
                    continue
                paths = dict(node.paths)
                paths[agent] = path
                child = _Node(constraints, paths)
                children.append(child)
                bounds.append(child.bound)
            classification = ConflictAssessment(
                node.bound, bounds,
            ).classification()
            split = _Split(conflict, classification, tuple(children))
            if (preferred is None or rank[classification]
                    < rank[preferred.classification]):
                preferred = split
            if classification == 'cardinal':
                break
        if preferred is None:
            raise ValueError('cannot split a conflict-free node')
        return preferred

    @staticmethod
    def _bypass(
        network: DroneNetwork, node: _Node, conflicts: list[Conflict],
        split: _Split, budget: SearchBudget,
    ) -> _Node | None:
        """Adopt helpful equal-arrival paths under parent constraints.

        Args:
            network (DroneNetwork): Topology used to count conflicts.
            node (_Node): Parent whose constraints remain in effect.
            conflicts (list[Conflict]): Parent's current conflict list.
            split (_Split): Tested split whose children may offer a bypass.
            budget (SearchBudget): Cooperative interruption controls.

        Returns:
            _Node | None: Parent-constraint node with fewer conflicts or,
                at equal conflict count, less total excess resource usage.
                None when no helpful equal-arrival, equal-cost path exists.

        Raises:
            PlanningCancelled: If cancellation is requested.
            SearchTimeout: If the budget deadline elapses.
        """
        if split.classification == 'cardinal':
            return None
        original = len(conflicts), sum(item.excess for item in conflicts)
        for child in split.children:
            budget.check()
            changed = next(agent for agent in split.conflict.agents
                           if child.paths[agent] is not node.paths[agent])
            if child.paths[changed].arrival != node.paths[changed].arrival:
                continue
            if child.paths[changed].cost != node.paths[changed].cost:
                continue
            remaining = ConflictDetector(network).find(child.paths)
            score = len(remaining), sum(item.excess for item in remaining)
            # Moving one drone off a heavily overloaded resource may leave
            # its conflict present. Strictly reducing excess still makes
            # progress, without restricting the parent's solution family.
            if score < original:
                return _Node(node.constraints, child.paths)
        return None
