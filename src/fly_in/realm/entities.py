from dataclasses import dataclass, field

from fly_in.models import ZoneType


@dataclass
class Zone:
    """A network hub; a None capacity denotes an unlimited endpoint."""

    name: str
    x: int
    y: int
    zone_type: ZoneType = ZoneType.NORMAL
    max_drones: int | None = 1

    def movement_cost(self) -> int:
        """Return the number of turns needed to enter this zone."""
        if self.zone_type is ZoneType.BLOCKED:
            raise ValueError(f"blocked zone '{self.name}' is inaccessible")
        if self.zone_type is ZoneType.RESTRICTED:
            return 2
        return 1

    def has_capacity(self, n: int) -> bool:
        """Return whether this zone can hold ``n`` drones."""
        return n >= 0 and (self.max_drones is None or n <= self.max_drones)


@dataclass
class Connection:
    """A bidirectional link with a shared per-turn traffic limit."""

    zone_a: Zone
    zone_b: Zone
    max_link_capacity: int = 1
    current_traffic: int = 0
    _identifier: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache the canonical name for this run's static topology."""
        self._identifier = '-'.join(sorted((
            self.zone_a.name, self.zone_b.name,
        )))

    def other_end(self, zone: Zone) -> Zone:
        """Return the opposite endpoint of this connection."""
        if zone is self.zone_a:
            return self.zone_b
        if zone is self.zone_b:
            return self.zone_a
        raise ValueError(f"zone '{zone.name}' is not on this connection")

    def identifier(self) -> str:
        """Return a stable, endpoint-order-independent connection name."""
        return self._identifier


@dataclass
class Drone:
    """An active or delivered drone and its discrete movement history."""

    id: str
    current_zone: Zone
    in_transit: Connection | None = None
    transit_remaining: int = 0
    move_log: list[str] = field(default_factory=list)
    delivered: bool = False
