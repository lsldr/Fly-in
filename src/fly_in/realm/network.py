from dataclasses import dataclass, field

from fly_in.models import MapConfig, ZoneType
from fly_in.realm.entities import Connection, Zone


@dataclass
class DroneNetwork:
    """Mutable zones and indexed undirected connections."""

    zones: dict[str, Zone]
    adjacency: dict[str, list[Connection]]
    start: Zone
    end: Zone
    connections: dict[str, Connection] = field(init=False)

    def __post_init__(self) -> None:
        """Index edges and reject inaccessible endpoints."""
        self.connections = {
            edge.identifier(): edge
            for edges in self.adjacency.values() for edge in edges
        }
        if any(zone.zone_type is ZoneType.BLOCKED
               for zone in (self.start, self.end)):
            raise ValueError("start and end hubs cannot be blocked")

    def neighbors(self, zone: Zone) -> list[Zone]:
        """Return zones directly connected to ``zone``."""
        return [
            connection.other_end(zone)
            for connection in self.adjacency.get(zone.name, [])
        ]

    def connection_between(self, zone_a: Zone, zone_b: Zone) -> Connection:
        """Return the connection joining two adjacent zones."""
        identifier = "-".join(sorted((zone_a.name, zone_b.name)))
        connection = self.connections.get(identifier)
        if connection is not None:
            return connection
        raise ValueError(
            f"no connection between '{zone_a.name}' and '{zone_b.name}'"
        )

    @classmethod
    def from_config(cls, config: MapConfig) -> "DroneNetwork":
        """Build a mutable network from a validated map configuration."""
        specs = [config.start, *config.zones, config.end]
        zones: dict[str, Zone] = {}
        for spec in specs:
            zones[spec.name] = Zone(
                name=spec.name, x=spec.x, y=spec.y,
                zone_type=spec.zone_type,
                max_drones=getattr(spec, "max_drones", None),
            )
        adjacency: dict[str, list[Connection]] = {
            name: [] for name in zones
        }
        for route in config.connections:
            connection = Connection(
                zones[route.from_hub], zones[route.to_hub],
                route.max_link_capacity,
            )
            adjacency[route.from_hub].append(connection)
            adjacency[route.to_hub].append(connection)
        return cls(
            zones, adjacency, zones[config.start.name],
            zones[config.end.name],
        )
