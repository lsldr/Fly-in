"""Shared capacity-aware space-time search and schedule execution primitives.

States are hub occupancy after a completed turn, with the initial state at 0.
A move departing state t claims its link at executed turn t+1 only. Restricted
arrival occurs at t+2 and releases that link slot for other departures.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Literal

from fly_in.models import ZoneType
from fly_in.realm.entities import Connection, Drone, Zone
from fly_in.realm.network import DroneNetwork
from fly_in.solvers.bounds import NetworkProfiler
from fly_in.solvers.protocol import Schedule

TimedState = tuple[str, int]
ResourceKind = Literal['zone', 'link']


class PlanningCancelled(RuntimeError):
    """Signal explicit cancellation rather than a search-budget cutoff."""


class SearchTimeout(RuntimeError):
    """Signal exhaustion of a cooperative search deadline."""


@dataclass(frozen=True)
class SearchBudget:
    """Share cancellation and the earliest deadline across nested searches."""

    deadline: float | None = None
    cancelled: Callable[[], bool] | None = None

    def check(self) -> None:
        """Raise before performing further work after cancellation/timeout."""
        if self.cancelled is not None and self.cancelled():
            raise PlanningCancelled('schedule preparation cancelled')
        if self.deadline is not None and perf_counter() >= self.deadline:
            raise SearchTimeout('planning time budget exhausted')

    def child(self, seconds: float | None) -> SearchBudget:
        """Cap a stage by both its own budget and its parent's deadline."""
        if seconds is None:
            return SearchBudget(self.deadline, self.cancelled)
        deadline = perf_counter() + max(0.0, seconds)
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        return SearchBudget(deadline, self.cancelled)

    def intersect(self, other: SearchBudget | None) -> SearchBudget:
        """Keep both callers' cancellation tokens and the earlier deadline."""
        if other is None or other is self:
            return self
        deadlines = [limit for limit in (self.deadline, other.deadline)
                     if limit is not None]

        def cancelled() -> bool:
            """Interrupt when either inherited caller requests cancellation."""
            return bool(
                (self.cancelled is not None and self.cancelled())
                or (other.cancelled is not None and other.cancelled())
            )

        return SearchBudget(min(deadlines) if deadlines else None, cancelled)


@dataclass(frozen=True)
class Constraint:
    """Forbid one agent from claiming one resource at a completed turn."""

    agent: str
    kind: ResourceKind
    resource: str
    time: int


@dataclass(frozen=True)
class TimedMove:
    """One traversal, expressed using source-state and arrival times."""

    source: str
    destination: str
    link: str
    departure: int
    arrival: int

    @property
    def duration(self) -> int:
        """Return physical entering-zone duration, excluding prior waiting."""
        return self.arrival - self.departure


@dataclass(frozen=True)
class TimedPath:
    """Immutable single-agent path including each occupied wait state."""

    agent: str
    states: tuple[TimedState, ...]
    moves: tuple[TimedMove, ...]

    @property
    def arrival(self) -> int:
        """Return the completed turn at which the drone is delivered."""
        return self.states[-1][1]

    @property
    def cost(self) -> int:
        """Sum physical traversal durations, excluding waiting."""
        return sum(move.duration for move in self.moves)

    def resources(
        self, network: DroneNetwork,
    ) -> Iterable[tuple[ResourceKind, str, int]]:
        """Yield constrained hub occupancy and link-entry resources."""
        for name, turn in self.states:
            if name not in (network.start.name, network.end.name):
                yield 'zone', name, turn
        for move in self.moves:
            yield 'link', move.link, move.departure + 1


@dataclass
class SearchStats:
    """Counters for comparison and profiling without machine-time claims."""

    expanded: int = 0
    nodes: int = 0
    iterations: int = 0
    accepted: int = 0
    bypasses: int = 0
    cardinal: int = 0
    semi_cardinal: int = 0
    non_cardinal: int = 0


@dataclass
class Solution:
    """Feasible paths or a search-node candidate, plus honest proof status."""

    paths: dict[str, TimedPath]
    optimal: bool = False
    status: str = 'feasible'
    stats: SearchStats = field(default_factory=SearchStats)
    reason: str | None = None
    lower_bound: int | None = None

    @property
    def makespan(self) -> int:
        """Return the latest delivery turn."""
        return max((path.arrival for path in self.paths.values()), default=0)

    @property
    def cost(self) -> int:
        """Return aggregate physical movement cost."""
        return sum(path.cost for path in self.paths.values())

    @property
    def objective(self) -> tuple[int, int]:
        """Order feasible solutions by makespan then physical path cost."""
        return self.makespan, self.cost

    def schedule(self) -> Schedule:
        """Produce departure-only rows consumed by Simulation.step()."""
        rows: Schedule = [[] for _ in range(self.makespan)]
        for name in sorted(self.paths, key=AgentOrder.key):
            for move in self.paths[name].moves:
                target = move.link if move.duration == 2 else move.destination
                rows[move.departure].append(f'{name}-{target}')
        return rows


class AgentOrder:
    """Central policy for numeric drone IDs and deterministic other IDs."""

    @staticmethod
    def key(name: str) -> tuple[int, int | str]:
        """Order D<number> numerically and other identifiers lexically.

        Equal numeric spellings retain their input order via stable sorting.
        This preserves the original nonnumeric lexical ordering policy.
        """
        suffix = name.removeprefix('D')
        return (0, int(suffix)) if suffix.isdecimal() else (1, name)

    @classmethod
    def ordered(cls, identifiers: Iterable[str]) -> list[str]:
        """Return identifiers in the shared deterministic planning order."""
        return sorted(identifiers, key=cls.key)


class ReservationTable:
    """Sparse counts with reversible reservations for destroy-and-repair."""

    def __init__(self, network: DroneNetwork) -> None:
        """Start with no occupancy claims on this topology."""
        self.network = network
        self._zone_use: Counter[TimedState] = Counter()
        self._link_use: Counter[TimedState] = Counter()

    def can_occupy(self, zone: Zone, turn: int) -> bool:
        """Check post-turn hub occupancy, including unlimited endpoints."""
        return zone.has_capacity(self._zone_use[zone.name, turn] + 1)

    def can_traverse(self, connection: Connection, turn: int) -> bool:
        """Check a link's entry-turn capacity, without arrival-turn claims."""
        return (self._link_use[connection.identifier(), turn]
                < connection.max_link_capacity)

    def reserve(self, path: TimedPath) -> None:
        """Claim each resource along a single-agent path."""
        for kind, name, turn in path.resources(self.network):
            table = self._zone_use if kind == 'zone' else self._link_use
            table[name, turn] += 1

    def unreserve(self, path: TimedPath) -> None:
        """Release a path atomically, rejecting unmatched removal."""
        claims = Counter(path.resources(self.network))
        for (kind, name, turn), amount in claims.items():
            table = self._zone_use if kind == 'zone' else self._link_use
            if table[name, turn] < amount:
                raise ValueError('cannot remove an unreserved path')
        for (kind, name, turn), amount in claims.items():
            table = self._zone_use if kind == 'zone' else self._link_use
            table[name, turn] -= amount
            if not table[name, turn]:
                del table[name, turn]


class SpaceTimeAStar:
    """A* over hub/time with reservations and agent-specific prohibitions."""

    def __init__(
        self, network: DroneNetwork, budget: SearchBudget | None = None,
    ) -> None:
        """Cache exact static reverse-Dijkstra durations once per engine."""
        self.network = network
        self.budget = budget or SearchBudget()
        self.expanded = 0
        self.priority_distances: dict[str, int] = {}
        self.distances = self._distances()
        self.transitions: dict[
            str, list[tuple[str, int, str, int | None, int, int]]
        ] = {}
        for name, zone in network.zones.items():
            steps = []
            for edge in network.adjacency[name]:
                target = edge.other_end(zone)
                if target.name in self.distances:
                    steps.append((
                        target.name, target.movement_cost(), edge.identifier(),
                        target.max_drones, edge.max_link_capacity,
                        int(target.zone_type is not ZoneType.PRIORITY),
                    ))
            self.transitions[name] = steps

    def _distances(self) -> dict[str, int]:
        """Compute entering-zone costs correctly in the reverse graph."""
        scores = NetworkProfiler(self.network, self.budget).route_scores(
            self.network.end.name, reverse=True,
        )
        self.priority_distances = {
            name: score[1] for name, score in scores.items()
        }
        return {name: score[0] for name, score in scores.items()}

    def find_path(
        self, agent: str, reservations: ReservationTable | None = None,
        constraints: Iterable[Constraint] = (), horizon: int = 100,
        budget: SearchBudget | None = None,
    ) -> TimedPath | None:
        """Find earliest arrival, then physical cost, then priority preference.

        Return None only for a bounded infeasible search; cancellation and
        elapsed deadlines have distinct exceptions so callers retain floors.
        """
        limits = self.budget.intersect(budget)
        limits.check()
        network = self.network
        start, goal = network.start.name, network.end.name
        if start not in self.distances:
            return None
        table = reservations or ReservationTable(network)
        forbidden = {(c.kind, c.resource, c.time) for c in constraints
                     if c.agent == agent}
        if ('zone', start, 0) in forbidden:
            return None
        serial = count()
        h = self.distances[start]
        queue = [(h, h, self.priority_distances[start], next(serial),
                  0, 0, 0, start)]
        best: dict[TimedState, tuple[int, int]] = {(start, 0): (0, 0)}
        previous: dict[TimedState, tuple[TimedState, TimedMove | None]] = {}
        while queue:
            limits.check()
            _, _, _, _, time, cost, ordinary, name = heappop(queue)
            state = name, time
            if best.get(state) != (cost, ordinary):
                continue
            self.expanded += 1
            if name == goal:
                states = [state]
                moves = []
                while state != (start, 0):
                    limits.check()
                    state, move = previous[state]
                    states.append(state)
                    if move is not None:
                        moves.append(move)
                return TimedPath(agent, tuple(reversed(states)),
                                 tuple(reversed(moves)))
            zone = network.zones[name]
            candidates: list[tuple[str, int, int, int, TimedMove | None]] = []
            if (time < horizon and table.can_occupy(zone, time+1)
                    and ('zone', name, time+1) not in forbidden):
                candidates.append((name, time+1, cost, ordinary, None))
            for target, duration, link, cap, link_cap, penalty in (
                self.transitions[name]
            ):
                arrival = time + duration
                score = cost+duration, ordinary+penalty
                state_key = target, arrival
                if state_key in best and best[state_key] <= score:
                    continue
                if (arrival + self.distances[target] > horizon
                        or table._link_use[link, time+1] >= link_cap
                        or (cap is not None
                            and table._zone_use[target, arrival] >= cap)
                        or ('link', link, time+1) in forbidden
                        or ('zone', target, arrival) in forbidden):
                    continue
                move = TimedMove(name, target, link, time, arrival)
                candidates.append((target, arrival, *score, move))
            for target_name, arrival, new_cost, penalty, move in candidates:
                target_state = target_name, arrival
                score = new_cost, penalty
                if target_state in best and best[target_state] <= score:
                    continue
                remaining = self.distances[target_name]
                if arrival + remaining > horizon:
                    continue
                best[target_state] = score
                previous[target_state] = ((name, time), move)
                heappush(queue, (
                    arrival+remaining, new_cost+remaining,
                    penalty+self.priority_distances[target_name],
                    next(serial), arrival, new_cost, penalty, target_name,
                ))
        return None


@dataclass(frozen=True)
class Conflict:
    """A minimal capacity+1 violating set at one resource and turn."""

    kind: ResourceKind
    resource: str
    time: int
    agents: tuple[str, ...]
    capacity: int

    def constraint(self, agent: str) -> Constraint:
        """Forbid this resource use for one involved agent."""
        return Constraint(agent, self.kind, self.resource, self.time)


class ConflictDetector:
    """Detect capacity violations on one bound, unchanged topology."""

    def __init__(self, network: DroneNetwork) -> None:
        """Bind resource lookups to the network for this preparation."""
        self.network = network

    def find(self, paths: Mapping[str, TimedPath]) -> list[Conflict]:
        """Find minimal violating groups, ignoring unlimited endpoints."""
        network = self.network
        resources: dict[tuple[ResourceKind, str, int], list[str]] = {}
        for agent in sorted(paths, key=AgentOrder.key):
            for resource in paths[agent].resources(network):
                resources.setdefault(resource, []).append(agent)
        conflicts = []
        for (kind, name, turn), agents in resources.items():
            if kind == 'zone':
                capacity = network.zones[name].max_drones
                if capacity is None:
                    continue
            else:
                capacity = network.connections[name].max_link_capacity
            if len(agents) > capacity:
                conflicts.append(Conflict(
                    kind, name, turn, tuple(agents[:capacity+1]), capacity,
                ))
        return sorted(conflicts, key=lambda item: (
            item.time, item.kind, item.resource, item.agents,
        ))


class SolutionValidator:
    """Establish fleet, physical-path and resource legality before proof."""

    def __init__(self, network: DroneNetwork) -> None:
        """Bind the topology used to validate an incumbent or final plan."""
        self.network = network
        self.detector = ConflictDetector(network)

    def validate(
        self, solution: Solution, drone_ids: list[str],
        budget: SearchBudget | None = None,
    ) -> None:
        """Reject incomplete, inconsistent, illegal or conflicting paths."""
        limits = budget or SearchBudget()
        limits.check()
        if (not drone_ids or len(set(drone_ids)) != len(drone_ids)
                or set(solution.paths) != set(drone_ids)):
            raise ValueError('solution must cover exactly the requested fleet')
        for agent, path in solution.paths.items():
            limits.check()
            if agent != path.agent:
                raise ValueError('path agent does not match its fleet key')
            self._validate_path(path, limits)
        if self.detector.find(solution.paths):
            raise ValueError('solution contains conflicting resource claims')
        limits.check()

    def _validate_path(self, path: TimedPath, budget: SearchBudget) -> None:
        """Require contiguous waits or physical traversals ending at goal."""
        network = self.network
        if (not path.states or path.states[0] != (network.start.name, 0)
                or path.states[-1][0] != network.end.name):
            raise ValueError('path must start at turn zero and end at goal')
        for index, (name, turn) in enumerate(path.states):
            budget.check()
            if (name not in network.zones or not isinstance(turn, int)
                    or turn < 0
                    or network.zones[name].zone_type is ZoneType.BLOCKED):
                raise ValueError('path has an invalid or blocked timed state')
            if name == network.end.name and index != len(path.states) - 1:
                raise ValueError('path continues after delivery')
        moves = {move.departure: move for move in path.moves}
        if len(moves) != len(path.moves):
            raise ValueError('path has duplicate departure turns')
        used: list[TimedMove] = []
        for (source, departure), (target, arrival) in zip(
            path.states, path.states[1:],
        ):
            budget.check()
            move = moves.get(departure)
            if move is None:
                if target != source or arrival != departure + 1:
                    raise ValueError(
                        'path has an unexplained timed transition'
                    )
                continue
            edge = network.connection_between(
                network.zones[source], network.zones[target],
            )
            if (move.source != source or move.destination != target
                    or move.arrival != arrival
                    or move.link != edge.identifier()
                    or arrival - departure
                    != network.zones[target].movement_cost()):
                raise ValueError('path move disagrees with its timed states')
            used.append(move)
        if tuple(used) != path.moves:
            raise ValueError('path moves must match its chronological states')


class ScheduledSolver:
    """Common prepare/step adapter for immutable complete solver schedules."""

    def __init__(self) -> None:
        """Start without a prepared solution or bound fleet."""
        self.solution: Solution | None = None
        self._schedule: Schedule = []
        self._next_turn = 0
        self._fleet: tuple[Drone, ...] = ()
        self._network: DroneNetwork | None = None

    def solve(
        self, network: DroneNetwork, drone_ids: list[str],
        initial: Solution | None = None, budget: SearchBudget | None = None,
    ) -> Solution:
        """Compute paths; concrete ladder algorithms override this method."""
        raise NotImplementedError

    def prepare(
        self, network: DroneNetwork, drones: list[Drone],
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Build one fresh fleet's schedule with cooperative cancellation."""
        self.solution = None
        self._schedule = []
        self._next_turn = 0
        self._fleet = ()
        self._network = None
        if not drones or not all(
            d.current_zone is network.start and d.in_transit is None
            and not d.delivered for d in drones
        ):
            raise ValueError('preparation requires a fresh fleet at start')
        result = self.solve(network, [d.id for d in drones],
                            budget=SearchBudget(cancelled=cancelled))
        SolutionValidator(network).validate(
            result, [d.id for d in drones], SearchBudget(cancelled=cancelled),
        )
        self.solution = result
        self._schedule = result.schedule()
        self._network = network
        self._fleet = tuple(drones)
        self._next_turn = 0

    @property
    def has_pending_actions(self) -> bool:
        """Allow legal all-wait turns before the cached plan ends."""
        return self._next_turn < len(self._schedule)

    def plan(self, network: DroneNetwork, drones: list[Drone]) -> Schedule:
        """Serve one cached departure row, preparing a new fleet if needed."""
        if (network is not self._network or len(drones) != len(self._fleet)
                or any(a is not b for a, b in zip(drones, self._fleet))):
            self.prepare(network, drones)
        if not self.has_pending_actions:
            return [[]]
        row = list(self._schedule[self._next_turn])
        self._next_turn += 1
        return [row]
