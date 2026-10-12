"""Mutable zones, links, and drones used to advance a simulation."""

from dataclasses import dataclass, field

from fly_in.models import ZoneType


@dataclass
class Zone:
    """A network hub; ``None`` capacity denotes an unlimited endpoint.

    Attributes:
        name (str): Unique hub identifier.
        x (int): Horizontal map coordinate.
        y (int): Vertical map coordinate.
        zone_type (ZoneType): Entry rule applied when a drone enters.
        max_drones (int | None): Occupancy limit, or ``None`` if unlimited.
    """

    name: str
    x: int
    y: int
    zone_type: ZoneType = ZoneType.NORMAL
    max_drones: int | None = 1

    def movement_cost(self) -> int:
        """Return the turns required to enter this zone.

        Returns:
            int: One for a normal or priority zone; two for a restricted
                zone.

        Raises:
            ValueError: If the zone is blocked and cannot be entered.
        """
        if self.zone_type is ZoneType.BLOCKED:
            raise ValueError(f"blocked zone '{self.name}' is inaccessible")
        if self.zone_type is ZoneType.RESTRICTED:
            return 2
        return 1

    def has_capacity(self, n: int) -> bool:
        """Check whether the zone can hold a proposed drone count.

        Args:
            n (int): Proposed number of drones; negative counts are rejected.

        Returns:
            bool: Whether ``n`` is nonnegative and within the configured
                limit.
        """
        return n >= 0 and (self.max_drones is None or n <= self.max_drones)


@dataclass
class Connection:
    """A bidirectional link with a shared per-turn traffic limit.

    Attributes:
        zone_a (Zone): One endpoint.
        zone_b (Zone): The other endpoint.
        max_link_capacity (int): Maximum traversals in one turn.
        current_traffic (int): Traversals currently charged this turn.
    """

    zone_a: Zone
    zone_b: Zone
    max_link_capacity: int = 1
    current_traffic: int = 0
    _identifier: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache the canonical, endpoint-order-independent link name."""
        self._identifier = '-'.join(sorted((
            self.zone_a.name, self.zone_b.name,
        )))

    def other_end(self, zone: Zone) -> Zone:
        """Return the endpoint opposite ``zone``.

        Args:
            zone (Zone): One endpoint of this connection, by object identity.

        Returns:
            Zone: The other endpoint.

        Raises:
            ValueError: If ``zone`` is not either endpoint.
        """
        if zone is self.zone_a:
            return self.zone_b
        if zone is self.zone_b:
            return self.zone_a
        raise ValueError(f"zone '{zone.name}' is not on this connection")

    def identifier(self) -> str:
        """Return the stable, endpoint-order-independent connection name.

        Returns:
            str: The two endpoint names joined by a hyphen in sorted order.
        """
        return self._identifier


@dataclass
class Drone:
    """An active or delivered drone and its discrete movement history.

    Attributes:
        id (str): Drone identifier used in action and event strings.
        current_zone (Zone): Current or last occupied endpoint.
        in_transit (Connection | None): Link being traversed, if any.
        transit_remaining (int): Remaining transit turns tracked by the run.
        move_log (list[str]): Movement events recorded for this drone.
        delivered (bool): Whether the drone has reached the network end.
    """

    id: str
    current_zone: Zone
    in_transit: Connection | None = None
    transit_remaining: int = 0
    move_log: list[str] = field(default_factory=list)
    delivered: bool = False
