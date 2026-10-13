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
    """Exception for explicit cancellation, distinct from a search cutoff."""


class SearchTimeout(RuntimeError):
    """Exception for exhaustion of a cooperative search deadline."""


@dataclass(frozen=True)
class SearchBudget:
    """Cancellation probe and earliest deadline shared across nested searches.

    Attributes:
        deadline (float | None): Absolute ``perf_counter`` deadline in seconds.
        cancelled (Callable[[], bool] | None): Optional cancellation probe.
    """

    deadline: float | None = None
    cancelled: Callable[[], bool] | None = None

    def check(self) -> None:
        """Raise before further work after cancellation or timeout.

        Raises:
            PlanningCancelled: If the cancellation probe returns true.
            SearchTimeout: If the absolute deadline has elapsed.
        """
        if self.cancelled is not None and self.cancelled():
            raise PlanningCancelled('schedule preparation cancelled')
        if self.deadline is not None and perf_counter() >= self.deadline:
            raise SearchTimeout('planning time budget exhausted')

    def child(self, seconds: float | None) -> SearchBudget:
        """Cap a stage by its own duration and the parent's deadline.

        Args:
            seconds (float | None): Maximum stage duration in seconds. A
                negative value produces an immediately expired child budget.

        Returns:
            SearchBudget: Budget sharing cancellation and the earlier deadline.
        """
        if seconds is None:
            return SearchBudget(self.deadline, self.cancelled)
        deadline = perf_counter() + max(0.0, seconds)
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        return SearchBudget(deadline, self.cancelled)

    def intersect(self, other: SearchBudget | None) -> SearchBudget:
        """Keep both cancellation tokens and the earlier deadline.

        Args:
            other (SearchBudget | None): Budget to combine with this one.

        Returns:
            SearchBudget: Combined budget, or this instance when no merge is
                needed.
        """
        if other is None or other is self:
            return self
        deadlines = [limit for limit in (self.deadline, other.deadline)
                     if limit is not None]

        def cancelled() -> bool:
            """Report cancellation requested by either parent budget.

            Returns:
                bool: Whether either cancellation probe returns true.
            """
            return bool(
                (self.cancelled is not None and self.cancelled())
                or (other.cancelled is not None and other.cancelled())
            )

        return SearchBudget(min(deadlines) if deadlines else None, cancelled)


@dataclass(frozen=True)
class Constraint:
    """Restriction on one agent's use of a resource at a completed turn.

    Attributes:
        agent (str): Drone identifier to constrain.
        kind (ResourceKind): Whether the resource is a hub or a link.
        resource (str): Hub name or connection identifier.
        time (int): Completed turn when use is forbidden.
    """

    agent: str
    kind: ResourceKind
    resource: str
    time: int


@dataclass(frozen=True)
class TimedMove:
    """One traversal, expressed using source-state and arrival times.

    Attributes:
        source (str): Hub occupied at departure.
        destination (str): Hub reached after traversal.
        link (str): Connection identifier used by the traversal.
        departure (int): Completed turn at the source before departure.
        arrival (int): Completed turn at the destination.
    """

    source: str
    destination: str
    link: str
    departure: int
    arrival: int

    @property
    def duration(self) -> int:
        """Physical entering-zone duration, excluding prior waiting.

        Returns:
            int: Arrival turn minus departure turn.
        """
        return self.arrival - self.departure


@dataclass(frozen=True)
class TimedPath:
    """Immutable single-agent path including each occupied wait state.

    Attributes:
        agent (str): Drone identifier.
        states (tuple[TimedState, ...]): Ordered hub and completed-turn states.
        moves (tuple[TimedMove, ...]): Traversals in chronological order.
    """

    agent: str
    states: tuple[TimedState, ...]
    moves: tuple[TimedMove, ...]

    @property
    def arrival(self) -> int:
        """Completed turn at which the drone is delivered.

        Returns:
            int: Arrival turn of the final state.
        """
        return self.states[-1][1]

    @property
    def cost(self) -> int:
        """Sum of physical traversal durations, excluding waiting.

        Returns:
            int: Total movement turns across all traversals.
        """
        return sum(move.duration for move in self.moves)

    def resources(
        self, network: DroneNetwork,
    ) -> Iterable[tuple[ResourceKind, str, int]]:
        """Yield constrained hub occupancy and link-entry resources.

        Args:
            network (DroneNetwork): Topology used to identify unlimited
                endpoints.

        Yields:
            tuple[ResourceKind, str, int]: Resource kind, identifier, and
                completed turn claimed by this path.
        """
        for name, turn in self.states:
            if name not in (network.start.name, network.end.name):
                yield 'zone', name, turn
        for move in self.moves:
            yield 'link', move.link, move.departure + 1


@dataclass
class SearchStats:
    """Counters for comparison and profiling without machine-time claims.

    Attributes:
        expanded (int): Low-level search states expanded.
        nodes (int): High-level search nodes processed.
        iterations (int): Improvement iterations attempted.
        accepted (int): Candidate schedules accepted by an improver.
        bypasses (int): ICBS bypasses applied.
        cardinal (int): Cardinal conflicts classified.
        semi_cardinal (int): Semi-cardinal conflicts classified.
        non_cardinal (int): Non-cardinal conflicts classified.
    """

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
    """Feasible paths or a search-node candidate, plus honest proof status.

    Attributes:
        paths (dict[str, TimedPath]): Paths keyed by drone identifier.
        optimal (bool): Whether minimum makespan has been proved.
        status (str): Solver outcome such as feasible, optimal, or timeout.
        stats (SearchStats): Search counters accumulated for this result.
        reason (str | None): Explanation for status or proof outcome.
        lower_bound (int | None): Certified lower bound on makespan, in turns.
    """

    paths: dict[str, TimedPath]
    optimal: bool = False
    status: str = 'feasible'
    stats: SearchStats = field(default_factory=SearchStats)
    reason: str | None = None
    lower_bound: int | None = None

    @property
    def makespan(self) -> int:
        """Latest delivery turn.

        Returns:
            int: Maximum path arrival, or zero for an empty mapping.
        """
        return max((path.arrival for path in self.paths.values()), default=0)

    @property
    def cost(self) -> int:
        """Aggregate physical movement cost.

        Returns:
            int: Sum of all drones' traversal durations, excluding waits.
        """
        return sum(path.cost for path in self.paths.values())

    @property
    def objective(self) -> tuple[int, int]:
        """Makespan and physical path cost in lexicographic order.

        Returns:
            tuple[int, int]: Makespan in turns and aggregate movement cost.
        """
        return self.makespan, self.cost

    def schedule(self) -> Schedule:
        """Produce departure-only rows consumed by ``Simulation.step``.

        Returns:
            Schedule: One row per makespan turn; an empty fleet has no rows.
        """
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

        Args:
            name (str): Drone identifier to order.

        Returns:
            tuple[int, int | str]: Sort key, with numeric IDs before others.
        """
        suffix = name.removeprefix('D')
        return (0, int(suffix)) if suffix.isdecimal() else (1, name)

    @classmethod
    def ordered(cls, identifiers: Iterable[str]) -> list[str]:
        """Return identifiers in the shared deterministic planning order.

        Args:
            identifiers (Iterable[str]): Drone identifiers to sort.

        Returns:
            list[str]: Identifiers in numeric-then-lexical order.
        """
        return sorted(identifiers, key=cls.key)


class ReservationTable:
    """Sparse counts with reversible reservations for destroy-and-repair.

    Attributes:
        network (DroneNetwork): Topology used to interpret path resources.
    """

    def __init__(self, network: DroneNetwork) -> None:
        """Start with no occupancy claims on this topology.

        Args:
            network (DroneNetwork): Topology whose capacities are reserved.
        """
        self.network = network
        self._zone_use: Counter[TimedState] = Counter()
        self._link_use: Counter[TimedState] = Counter()

    def can_occupy(self, zone: Zone, turn: int) -> bool:
        """Check post-turn hub occupancy, including unlimited endpoints.

        Args:
            zone (Zone): Hub whose capacity is checked.
            turn (int): Completed turn to inspect.

        Returns:
            bool: Whether one more drone can occupy the hub at that turn.
        """
        return zone.has_capacity(self._zone_use[zone.name, turn] + 1)

    def can_traverse(self, connection: Connection, turn: int) -> bool:
        """Check link entry capacity, without arrival-turn claims.

        Args:
            connection (Connection): Link whose capacity is checked.
            turn (int): Completed turn on which the link is entered.

        Returns:
            bool: Whether another traversal can claim this link and turn.
        """
        return (self._link_use[connection.identifier(), turn]
                < connection.max_link_capacity)

    def reserve(self, path: TimedPath) -> None:
        """Claim each resource along a single-agent path.

        Args:
            path (TimedPath): Path whose constrained uses are added.
        """
        for kind, name, turn in path.resources(self.network):
            table = self._zone_use if kind == 'zone' else self._link_use
            table[name, turn] += 1

    def unreserve(self, path: TimedPath) -> None:
        """Release a path atomically, rejecting unmatched removal.

        Args:
            path (TimedPath): Previously reserved path to release.

        Raises:
            ValueError: If any path claim is absent from the table.
        """
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
    """A* over hub/time with reservations and agent-specific prohibitions.

    Attributes:
        network (DroneNetwork): Topology searched by this engine.
        budget (SearchBudget): Default cooperative cancellation and deadline.
        expanded (int): Number of low-level states expanded so far.
        priority_distances (dict[str, int]): Secondary route tie-break values.
        distances (dict[str, int]): Exact static movement costs to the goal.
        transitions (dict[str, list[tuple]]): Reachable outgoing movements.
    """

    def __init__(
        self, network: DroneNetwork, budget: SearchBudget | None = None,
    ) -> None:
        """Cache exact static reverse-Dijkstra durations once per engine.

        Args:
            network (DroneNetwork): Immutable topology to search.
            budget (SearchBudget | None): Default limits for nested searches.
        """
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
        """Compute entering-zone costs correctly in the reverse graph.

        Returns:
            dict[str, int]: Minimum physical movement turns from each reachable
                hub to the goal.
        """
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

        Args:
            agent (str): Drone identifier associated with the returned path.
            reservations (ReservationTable | None): Existing resource claims.
            constraints (Iterable[Constraint]): Agent-specific forbidden uses.
            horizon (int): Latest permitted arrival turn, inclusive.
            budget (SearchBudget | None): Additional limits for this search.

        Returns:
            TimedPath | None: Earliest feasible path under the horizon, or
                None when no path fits.

        Raises:
            PlanningCancelled: If an active caller requests cancellation.
            SearchTimeout: If an active deadline elapses.
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
    """A minimal capacity+1 violating set at one resource and turn.

    Attributes:
        kind (ResourceKind): Whether the overused resource is a hub or link.
        resource (str): Hub name or connection identifier.
        time (int): Completed turn of the capacity violation.
        agents (tuple[str, ...]): First capacity-plus-one users in stable
            order.
        capacity (int): Maximum concurrent use of the resource.
        excess (int): Total users above capacity, including users outside
            the minimal violating set used for branching.
    """

    kind: ResourceKind
    resource: str
    time: int
    agents: tuple[str, ...]
    capacity: int
    excess: int = 1

    def constraint(self, agent: str) -> Constraint:
        """Forbid this resource use for one involved agent.

        Args:
            agent (str): Drone to exclude from this use.

        Returns:
            Constraint: Restriction matching this conflict's resource and turn.
        """
        return Constraint(agent, self.kind, self.resource, self.time)


class ConflictDetector:
    """Capacity-violation detector bound to one unchanged topology.

    Attributes:
        network (DroneNetwork): Topology supplying capacities and endpoints.
    """

    def __init__(self, network: DroneNetwork) -> None:
        """Bind resource lookups to the network for this preparation.

        Args:
            network (DroneNetwork): Topology to inspect.
        """
        self.network = network

    def find(self, paths: Mapping[str, TimedPath]) -> list[Conflict]:
        """Find minimal violating groups, ignoring unlimited endpoints.

        Args:
            paths (Mapping[str, TimedPath]): Candidate paths keyed by drone.

        Returns:
            list[Conflict]: Capacity violations ordered by turn and resource.
        """
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
                    len(agents) - capacity,
                ))
        return sorted(conflicts, key=lambda item: (
            item.time, item.kind, item.resource, item.agents,
        ))


class SolutionValidator:
    """Fleet, physical-path, and resource legality checker used before proof.

    Attributes:
        network (DroneNetwork): Topology against which paths are checked.
        detector (ConflictDetector): Capacity checker bound to that topology.
    """

    def __init__(self, network: DroneNetwork) -> None:
        """Bind the topology used to validate a plan.

        Args:
            network (DroneNetwork): Topology against which paths are checked.
        """
        self.network = network
        self.detector = ConflictDetector(network)

    def validate(
        self, solution: Solution, drone_ids: list[str],
        budget: SearchBudget | None = None,
    ) -> None:
        """Reject incomplete, inconsistent, illegal or conflicting paths.

        Args:
            solution (Solution): Candidate result to validate.
            drone_ids (list[str]): Exact fleet that the result must cover.
            budget (SearchBudget | None): Optional interruption controls.

        Raises:
            ValueError: If the fleet, paths, movements, or resource claims are
                invalid.
            PlanningCancelled: If the caller requests cancellation.
            SearchTimeout: If the supplied deadline elapses.
        """
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
        """Require contiguous waits or physical traversals ending at goal.

        Args:
            path (TimedPath): Path to check against the bound network.
            budget (SearchBudget): Interruption controls for long paths.

        Raises:
            ValueError: If a state, wait, traversal, or chronology is invalid.
            PlanningCancelled: If the caller requests cancellation.
            SearchTimeout: If the supplied deadline elapses.
        """
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
    """Common prepare/step adapter for immutable complete solver schedules.

    Attributes:
        solution (Solution | None): Validated solution for the prepared fleet.
        has_pending_actions (bool): Whether a cached row remains to serve.
    """

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
        """Compute paths; concrete ladder algorithms override this method.

        Args:
            network (DroneNetwork): Topology for the requested plan.
            drone_ids (list[str]): Fleet identifiers to route.
            initial (Solution | None): Optional feasible incumbent.
            budget (SearchBudget | None): Optional deadline and cancellation.

        Returns:
            Solution: Candidate paths and their proof/status metadata.

        Raises:
            NotImplementedError: The base class does not implement planning.
        """
        raise NotImplementedError

    def prepare(
        self, network: DroneNetwork, drones: list[Drone],
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Build one fresh fleet's schedule with cooperative cancellation.

        Args:
            network (DroneNetwork): Topology for the plan.
            drones (list[Drone]): Fresh fleet at the start hub.
            cancelled (Callable[[], bool] | None): Optional cancellation probe.

        Raises:
            ValueError: If the fleet is empty or not fresh at the start.
            PlanningCancelled: If cancellation is requested during planning.
            SearchTimeout: If a concrete solver's deadline elapses.
        """
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
        """Whether the cached plan has an action row left to serve.

        Returns:
            bool: Whether another scheduled turn remains.
        """
        return self._next_turn < len(self._schedule)

    def plan(self, network: DroneNetwork, drones: list[Drone]) -> Schedule:
        """Serve one cached departure row, preparing a new fleet if needed.

        Args:
            network (DroneNetwork): Current topology.
            drones (list[Drone]): Current fleet objects.

        Returns:
            Schedule: A single row of departures, or one empty row after the
                cached plan has ended.
        """
        if (network is not self._network or len(drones) != len(self._fleet)
                or any(a is not b for a, b in zip(drones, self._fleet))):
            self.prepare(network, drones)
        if not self.has_pending_actions:
            return [[]]
        row = list(self._schedule[self._next_turn])
        self._next_turn += 1
        return [row]
