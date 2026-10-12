"""Mutable graph topology built for an individual simulation."""

from dataclasses import dataclass, field

from fly_in.models import MapConfig, ZoneType
from fly_in.realm.entities import Connection, Zone


@dataclass
class DroneNetwork:
    """Mutable zones and indexed undirected connections.

    Attributes:
        zones (dict[str, Zone]): Zones keyed by unique name.
        adjacency (dict[str, list[Connection]]): Incident links by zone name.
        start (Zone): Source endpoint.
        end (Zone): Destination endpoint.
        connections (dict[str, Connection]): Links keyed by canonical name.
    """

    zones: dict[str, Zone]
    adjacency: dict[str, list[Connection]]
    start: Zone
    end: Zone
    connections: dict[str, Connection] = field(init=False)

    def __post_init__(self) -> None:
        """Index links by name and reject blocked endpoints.

        Raises:
            ValueError: If the start or end zone is blocked.
        """
        self.connections = {
            edge.identifier(): edge
            for edges in self.adjacency.values() for edge in edges
        }
        if any(zone.zone_type is ZoneType.BLOCKED
               for zone in (self.start, self.end)):
            raise ValueError("start and end hubs cannot be blocked")

    def neighbors(self, zone: Zone) -> list[Zone]:
        """Return zones directly connected to ``zone``.

        Args:
            zone (Zone): Zone whose adjacent endpoints are requested.

        Returns:
            list[Zone]: Neighbor zones in adjacency-list order.
        """
        return [
            connection.other_end(zone)
            for connection in self.adjacency.get(zone.name, [])
        ]

    def connection_between(self, zone_a: Zone, zone_b: Zone) -> Connection:
        """Find the link joining two zones.

        Args:
            zone_a (Zone): First endpoint.
            zone_b (Zone): Second endpoint.

        Returns:
            Connection: Link indexed by the two endpoint names.

        Raises:
            ValueError: If no indexed link joins the endpoint names.
        """
        identifier = "-".join(sorted((zone_a.name, zone_b.name)))
        connection = self.connections.get(identifier)
        if connection is not None:
            return connection
        raise ValueError(
            f"no connection between '{zone_a.name}' and '{zone_b.name}'"
        )

    @classmethod
    def from_config(cls, config: MapConfig) -> "DroneNetwork":
        """Build a mutable network from a validated map configuration.

        Args:
            config (MapConfig): Validated map description.

        Returns:
            DroneNetwork: Fresh zones, adjacency lists, and indexed links.
        """
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
