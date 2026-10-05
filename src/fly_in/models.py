from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ZoneType(str, Enum):
    """Enumerate entering-zone movement rules."""

    NORMAL = "normal"
    BLOCKED = "blocked"
    RESTRICTED = "restricted"
    PRIORITY = "priority"


class ZoneSpec(BaseModel):
    """Shared validated properties of a zone."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    x: int
    y: int
    zone_type: ZoneType = ZoneType.NORMAL
    color: str | None = None

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        """Reject empty, dashed, or whitespace-containing hub names."""
        if "-" in value or any(character.isspace() for character in value):
            raise ValueError("zone names cannot contain dashes or whitespace")
        return value

    @field_validator("color")
    @classmethod
    def valid_color(cls, value: str | None) -> str | None:
        """Accept any nonempty single-word color value."""
        if value is not None and (
            not value or any(c.isspace() for c in value)
        ):
            raise ValueError("color must be a non-empty single word")
        return value


class RegZoneSpec(ZoneSpec):
    """A regular zone with a positive simultaneous-drone capacity."""

    max_drones: int = Field(default=1, gt=0)


class EdgeZoneSpec(ZoneSpec):
    """A start or end zone, which has no occupancy limit."""


class RouteSpec(BaseModel):
    """A bidirectional connection between two zones."""

    model_config = ConfigDict(frozen=True)

    from_hub: str
    to_hub: str
    max_link_capacity: int = Field(default=1, gt=0)

    @field_validator("from_hub", "to_hub")
    @classmethod
    def valid_hub_name(cls, value: str) -> str:
        """Validate an endpoint identifier without graph lookups."""
        if not value or "-" in value or any(c.isspace() for c in value):
            raise ValueError(
                "connection zone names cannot contain dashes or whitespace"
            )
        return value

    @model_validator(mode="after")
    def different_endpoints(self) -> "RouteSpec":
        """Reject connections that lead back to the same hub."""
        if self.from_hub == self.to_hub:
            raise ValueError("a connection must join two different zones")
        return self

    @property
    def name(self) -> str:
        """Return the conventional connection name."""
        return f"{self.from_hub}-{self.to_hub}"


class MapConfig(BaseModel):
    """A complete and internally consistent simulation map."""

    model_config = ConfigDict(frozen=True)

    nb_drones: int = Field(gt=0)
    start: EdgeZoneSpec
    end: EdgeZoneSpec
    zones: list[RegZoneSpec]
    connections: list[RouteSpec]

    @model_validator(mode="after")
    def validate_map(self) -> "MapConfig":
        """Validate unique hub names and undirected connections."""
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
