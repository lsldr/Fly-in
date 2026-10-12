"""Advance drone plans under simultaneous movement and capacity rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from fly_in.metrics import MetricsCollector, RunMetrics
from fly_in.models import MapConfig
from fly_in.realm.entities import Drone, Zone
from fly_in.realm.network import DroneNetwork
from fly_in.solvers.protocol import Solver


@dataclass(frozen=True)
class SimulationResult:
    """Completed run data, including formatted turn history.

    Attributes:
        turns (int): Absolute simulation turn at completion.
        history (list[list[str]]): Copied movement events grouped by turn.
        metrics (RunMetrics): Aggregate movement and delivery measurements.
    """

    turns: int
    history: list[list[str]]
    metrics: RunMetrics


@dataclass
class Simulation:
    """Mutable run state that applies actions under simultaneous capacities.

    Attributes:
        network (DroneNetwork): Mutable map topology for this run.
        drones (list[Drone]): Drone fleet and current state.
        solver (Solver): Strategy supplying actions for each turn.
        metrics (MetricsCollector): Measurements accumulated during turns.
        turn (int): Number of turns already completed.
        history (list[list[str]]): Movement events grouped by completed turn.
    """

    network: DroneNetwork
    drones: list[Drone]
    solver: Solver
    metrics: MetricsCollector
    turn: int = 0
    history: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: MapConfig, solver: Solver) -> "Simulation":
        """Create a simulation with a fresh network and drone fleet.

        Args:
            config (MapConfig): Validated map and fleet size.
            solver (Solver): Strategy used to plan movement.

        Returns:
            Simulation: Run initialized at the map's start endpoint.
        """
        network = DroneNetwork.from_config(config)
        drones = [
            Drone(f"D{index}", network.start)
            for index in range(1, config.nb_drones + 1)
        ]
        return cls(network, drones, solver, MetricsCollector())

    def run(
        self,
        on_turn: Callable[[Simulation, list[str]], None] | None = None,
        *, max_turns: int | None = None,
    ) -> SimulationResult:
        """Run until delivery, optionally limiting the absolute turn count.

        Args:
            on_turn (Callable[[Simulation, list[str]], None] | None): Optional
                callback invoked after each committed turn with copied events.
            max_turns (int | None): Optional positive absolute turn ceiling.
                Exceptions raised by solver planning propagate to the caller.

        Returns:
            SimulationResult: Completed turn count, copied history, and
                metric snapshot.

        Raises:
            ValueError: If ``max_turns`` is less than one.
            RuntimeError: If the limit is reached before delivery or no
                actions can progress the undelivered fleet.
            PlanningCancelled: If a solver's cancellation check stops
                planning.
            SearchTimeout: If a solver exhausts its search deadline, including
                when an optimal result cannot be proved.
        """
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be positive")
        while not all(drone.delivered for drone in self.drones):
            if max_turns is not None and self.turn >= max_turns:
                raise RuntimeError(
                    f"simulation reached the {max_turns}-turn limit"
                )
            moves = self.step()
            if (not moves
                and not getattr(self.solver, "has_pending_actions", False)
                and not any(drone.in_transit is not None
                            for drone in self.drones)):
                waiting = sum(not drone.delivered for drone in self.drones)
                raise RuntimeError(
                    f"simulation is deadlocked with {waiting} undelivered "
                    "drone(s)"
                )
            if on_turn is not None:
                on_turn(self, list(moves))
        return SimulationResult(self.turn, [list(row) for row in self.history],
                                self.metrics.summary())

    def step(self) -> list[str]:
        """Validate and commit one turn, then return its movement events.

        The full proposed action row and capacity constraints are checked
        before simulation state changes. A committed turn updates forced
        restricted-link arrivals, drone state, movement history, and metrics.
        Planning exceptions propagate without wrapping. If every drone is
        already delivered, return an empty list without advancing the turn.

        Returns:
            list[str]: Events for departures and arrivals committed this turn.

        Raises:
            ValueError: If the solver returns malformed, unavailable, or
                capacity-violating movements.
            PlanningCancelled: If solver planning is cancelled.
            SearchTimeout: If solver planning exhausts its time budget.
        """
        if all(drone.delivered for drone in self.drones):
            return []
        arriving_transit = [
            drone for drone in self.drones if drone.in_transit is not None
        ]
        plan = self.solver.plan(self.network, self.drones)
        actions = plan[0] if plan else []
        by_id = {drone.id: drone for drone in self.drones}
        departures: list[tuple[Drone, Zone, bool]] = []
        seen: set[str] = set()
        for action in actions:
            drone_id, separator, destination_name = action.partition("-")
            if not separator or drone_id not in by_id:
                raise ValueError(
                    f"solver returned invalid movement '{action}'"
                )
            if drone_id in seen:
                raise ValueError(f"duplicate action for drone '{drone_id}'")
            seen.add(drone_id)
            drone = by_id[drone_id]
            if drone.delivered or drone.in_transit is not None:
                raise ValueError(
                    f"solver moved unavailable drone '{drone_id}'"
                )
            source = drone.current_zone
            destination = self.network.zones.get(destination_name)
            restricted = False
            if destination is None:
                restricted_connection = next(
                    (candidate for candidate in
                     self.network.adjacency.get(source.name, [])
                     if candidate.identifier() == destination_name),
                    None,
                )
                if restricted_connection is None:
                    raise ValueError(
                        f"solver moved '{drone_id}' to unknown "
                        f"'{destination_name}'"
                    )
                destination = restricted_connection.other_end(source)
                restricted = True
            else:
                self.network.connection_between(source, destination)
            if restricted != (destination.movement_cost() == 2):
                raise ValueError(
                    f"movement for '{drone_id}' does not match destination "
                    "movement cost"
                )
            departures.append((drone, destination, restricted))

        self._validate_capacity(departures)
        for connection in self.network.connections.values():
            connection.current_traffic = 0
        moves: list[str] = []
        for drone, destination_obj, restricted in departures:
            destination = destination_obj
            connection = self.network.connection_between(
                drone.current_zone, destination
            )
            connection.current_traffic += 1
            self.metrics.record_path_cost(destination.movement_cost())
            if restricted:
                drone.in_transit = connection
                drone.transit_remaining = 1
                event = f"{drone.id}-{connection.identifier()}"
            else:
                drone.current_zone = destination
                drone.delivered = destination is self.network.end
                event = f"{drone.id}-{destination.name}"
            drone.move_log.append(event)
            moves.append(event)

        for drone in arriving_transit:
            if drone.in_transit is None:
                continue
            destination = drone.in_transit.other_end(drone.current_zone)
            drone.current_zone = destination
            drone.in_transit = None
            drone.transit_remaining = 0
            drone.delivered = destination is self.network.end
            event = f"{drone.id}-{destination.name}"
            drone.move_log.append(event)
            moves.append(event)

        self.turn += 1
        for drone in self.drones:
            if drone.delivered:
                self.metrics.record_arrival(drone.id, self.turn)
        self.metrics.record_turn(moves)
        self.history.append(list(moves))
        return moves

    def _validate_capacity(
        self, departures: list[tuple[Drone, Zone, bool]]
    ) -> None:
        """Reject departures that exceed link or destination capacity.

        Args:
            departures (list[tuple[Drone, Zone, bool]]): Proposed drone,
                destination, and restricted-traversal flag tuples.

        Raises:
            ValueError: If a connection traffic limit or zone occupancy limit
                would be exceeded.
        """
        leaving: dict[str, int] = {}
        arriving: dict[str, int] = {}
        link_counts: dict[str, int] = {}
        for drone in self.drones:
            if drone.delivered or drone.in_transit is not None:
                continue
            name = drone.current_zone.name
            leaving.setdefault(name, 0)
        for drone, destination_obj, restricted in departures:
            destination = destination_obj
            leaving[drone.current_zone.name] = (
                leaving.get(drone.current_zone.name, 0) + 1
            )
            if not restricted:
                arriving[destination.name] = (
                    arriving.get(destination.name, 0) + 1
                )
            connection = self.network.connection_between(
                drone.current_zone, destination
            )
            link_counts[connection.identifier()] = (
                link_counts.get(connection.identifier(), 0) + 1
            )
        for drone in self.drones:
            if drone.in_transit is not None:
                link = drone.in_transit
                destination = link.other_end(drone.current_zone)
                arriving[destination.name] = (
                    arriving.get(destination.name, 0) + 1
                )
        for name, count in link_counts.items():
            link = self.network.connections[name]
            if count > link.max_link_capacity:
                raise ValueError(f"connection '{name}' capacity exceeded")
        occupancy: dict[str, int] = {}
        for drone in self.drones:
            if not drone.delivered and drone.in_transit is None:
                name = drone.current_zone.name
                occupancy[name] = occupancy.get(name, 0) + 1
        for name, incoming in arriving.items():
            zone = self.network.zones[name]
            if zone is self.network.start or zone is self.network.end:
                continue
            final_count = (
                occupancy.get(name, 0) - leaving.get(name, 0) + incoming
            )
            if not zone.has_capacity(final_count):
                raise ValueError(f"zone '{name}' capacity exceeded")
