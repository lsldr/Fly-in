"""Validated immutable map configuration models."""

from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ZoneType(str, Enum):
    """Enumerate zone entry behavior and movement duration."""

    NORMAL = "normal"
    BLOCKED = "blocked"
    RESTRICTED = "restricted"
    PRIORITY = "priority"


class ZoneSpec(BaseModel):
    """Shared validated properties of a zone.

    Attributes:
        name (str): Nonempty hub name without dashes or whitespace.
        x (int): Horizontal map coordinate.
        y (int): Vertical map coordinate.
        zone_type (ZoneType): Entry behavior, defaulting to normal.
        color (str | None): Optional nonempty, single-word display color.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    x: int
    y: int
    zone_type: ZoneType = ZoneType.NORMAL
    color: str | None = None

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        """Validate a hub name for use in movement identifiers.

        Args:
            value (str): Candidate hub name.

        Returns:
            str: The unchanged name when valid.

        Raises:
            ValueError: If the name contains a dash or whitespace.
        """
        if "-" in value or any(character.isspace() for character in value):
            raise ValueError("zone names cannot contain dashes or whitespace")
        return value

    @field_validator("color")
    @classmethod
    def valid_color(cls, value: str | None) -> str | None:
        """Validate an optional single-word color value.

        Args:
            value (str | None): Color label, or ``None`` when unspecified.

        Returns:
            str | None: The unchanged color value.

        Raises:
            ValueError: If a supplied color is empty or contains whitespace.
        """
        if value is not None and (
            not value or any(c.isspace() for c in value)
        ):
            raise ValueError("color must be a non-empty single word")
        return value


class RegZoneSpec(ZoneSpec):
    """A regular zone with a positive simultaneous-drone capacity.

    Attributes:
        max_drones (int): Maximum number of drones allowed in the zone.
    """

    max_drones: int = Field(default=1, gt=0)


class EdgeZoneSpec(ZoneSpec):
    """A start or end zone, which has no occupancy limit."""


class RouteSpec(BaseModel):
    """A bidirectional connection between two zones.

    Attributes:
        from_hub (str): Name of one endpoint.
        to_hub (str): Name of the other endpoint.
        max_link_capacity (int): Positive simultaneous traversal limit.
    """

    model_config = ConfigDict(frozen=True)

    from_hub: str
    to_hub: str
    max_link_capacity: int = Field(default=1, gt=0)

    @field_validator("from_hub", "to_hub")
    @classmethod
    def valid_hub_name(cls, value: str) -> str:
        """Validate an endpoint name without checking graph membership.

        Args:
            value (str): Candidate endpoint identifier.

        Returns:
            str: The unchanged identifier when valid.

        Raises:
            ValueError: If the name is empty or contains a dash or whitespace.
        """
        if not value or "-" in value or any(c.isspace() for c in value):
            raise ValueError(
                "connection zone names cannot contain dashes or whitespace"
            )
        return value

    @model_validator(mode="after")
    def different_endpoints(self) -> "RouteSpec":
        """Require the connection to join distinct hubs.

        Returns:
            RouteSpec: This validated connection specification.

        Raises:
            ValueError: If both endpoint names are equal.
        """
        if self.from_hub == self.to_hub:
            raise ValueError("a connection must join two different zones")
        return self

    @property
    def name(self) -> str:
        """Conventional endpoint name joined by a hyphen.

        Returns:
            str: ``from_hub`` followed by ``to_hub`` separated by ``-``.
        """
        return f"{self.from_hub}-{self.to_hub}"


class MapConfig(BaseModel):
    """A complete and internally consistent simulation map.

    Attributes:
        nb_drones (int): Positive number of drones in the initial fleet.
        start (EdgeZoneSpec): Unlimited source endpoint.
        end (EdgeZoneSpec): Unlimited destination endpoint.
        zones (list[RegZoneSpec]): Capacity-limited intermediate zones.
        connections (list[RouteSpec]): Unique undirected map connections.
    """

    model_config = ConfigDict(frozen=True)

    nb_drones: int = Field(gt=0)
    start: EdgeZoneSpec
    end: EdgeZoneSpec
    zones: list[RegZoneSpec]
    connections: list[RouteSpec]

    @model_validator(mode="after")
    def validate_map(self) -> "MapConfig":
        """Validate unique hubs and valid, nonrepeated connections.

        Returns:
            MapConfig: This validated map.

        Raises:
            ValueError: If hub names repeat, a connection references an
                unknown hub, or an undirected connection is repeated.
        """
        zones = [self.start, *self.zones, self.end]
        names = [zone.name for zone in zones]
        if len(names) != len(set(names)):
            raise ValueError(
                "zone names must be unique, including start and end"
            )

        known_names = set(names)
        seen_connections: set[frozenset[str]] = set()
        for connection in self.connections:
            endpoints = {connection.from_hub, connection.to_hub}
            if not endpoints <= known_names:
                unknown = sorted(endpoints - known_names)[0]
                raise ValueError(
                    f"connection references undefined zone '{unknown}'"
                )
            pair = frozenset(endpoints)
            if pair in seen_connections:
                raise ValueError(
                    "connections cannot repeat the same zone pair"
                )
            seen_connections.add(pair)
        return self
