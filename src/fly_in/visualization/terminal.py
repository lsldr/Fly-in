"""Colored terminal rendering for simulation turns and occupancy state."""

from rich.color import Color, ColorParseError
from rich.console import Console
from rich.text import Text

from fly_in.models import MapConfig, RegZoneSpec, ZoneSpec, ZoneType
from fly_in.simulation import Simulation
from fly_in.visualization.terminal_scene import TerminalScene


class TerminalVisualizer:
    """Terminal output renderer for movement and network capacity state.

    Attributes:
        display (str): Output style: visual, compact, full, or moves.
        config (MapConfig): Topology supplying colors and route endpoints.
        console (Console): Rich console used for formatted output.
        scene (TerminalScene): Renderer for the coordinate-based network.
    """

    _TYPE_COLORS = {
        ZoneType.NORMAL: "#7aa2f7",
        ZoneType.BLOCKED: "#565f89",
        ZoneType.RESTRICTED: "#ff9e64",
        ZoneType.PRIORITY: "#bb9af7",
    }

    def __init__(
        self, config: MapConfig, no_color: bool = False,
        display: str = "visual",
    ) -> None:
        """Create a renderer using visual properties from ``config``.

        Args:
            config (MapConfig): Map whose zones and links are rendered.
            no_color (bool): Disable color markup in terminal output.
            display (str): Output mode: ``visual``, ``compact``, ``full``, or
                ``moves``.

        Raises:
            ValueError: If ``display`` is not a supported output mode.
        """
        if display not in {"visual", "compact", "full", "moves"}:
            raise ValueError("display must be visual, compact, full, or moves")
        self.display = display
        self.config = config
        self.console = Console(no_color=no_color, soft_wrap=True)
        self.scene = TerminalScene(config)
        self._zones = {
            zone.name: zone
            for zone in [config.start, *config.zones, config.end]
        }

    def render_turn(self, simulation: Simulation, moves: list[str]) -> None:
        """Print one movement line and the selected network summary.

        Args:
            simulation (Simulation): Post-turn state used for counts and
                traffic.
            moves (list[str]): Movement tokens produced for this turn.
        """
        if self.display == "moves":
            print(" ".join(moves))
            return
        if self.display == "visual":
            self.console.print(
                self.scene.render(simulation, moves, self.console.width),
                soft_wrap=False,
            )
            return
        movement_line = Text()
        for index, movement in enumerate(moves):
            if index:
                movement_line.append(" ")
            drone_id, _, target = movement.partition("-")
            movement_line.append(f"{drone_id}-", style="bold")
            movement_line.append(target, style=self._movement_color(target))
        self.console.print(movement_line, markup=False, highlight=False)
        if self.display == "full":
            self.console.print(self._state_line(simulation), highlight=False)
        else:
            delivered = sum(drone.delivered for drone in simulation.drones)
            transit = sum(
                drone.in_transit is not None for drone in simulation.drones
            )
            self.console.print(
                f"Turn {simulation.turn} · delivered "
                f"{delivered}/{len(simulation.drones)} · "
                f"in transit {transit}", highlight=False,
            )

    def _movement_color(self, target: str) -> str:
        """Find the target zone's configured or default display color.

        Args:
            target (str): Zone name or canonical connection identifier.

        Returns:
            str: Rich color string for the zone or a restricted route endpoint;
            white when the target does not identify known topology.
        """
        zone = self._zones.get(target)
        if zone is None:
            connection = next(
                (
                    item
                    for item in self.config.connections
                    if "-".join(sorted((item.from_hub, item.to_hub)))
                    == target
                ),
                None,
            )
            if connection is None:
                return "white"
            endpoints = [
                self._zones[connection.from_hub],
                self._zones[connection.to_hub],
            ]
            zone = next(
                (
                    endpoint
                    for endpoint in endpoints
                    if endpoint.zone_type is ZoneType.RESTRICTED
                ),
                endpoints[1],
            )
        return self._zone_color(zone)

    def _state_line(self, simulation: Simulation) -> Text:
        """Build a colored summary of zone occupancy and link traffic.

        Args:
            simulation (Simulation): State whose current zone counts and link
                traffic are summarized.

        Returns:
            Text: Rich summary of zone occupancy and link capacity use.
        """
        occupancy: dict[str, int] = {}
        for drone in simulation.drones:
            if drone.delivered or drone.in_transit is not None:
                continue
            zone_name = drone.current_zone.name
            occupancy[zone_name] = occupancy.get(zone_name, 0) + 1

        line = Text(f"Turn {simulation.turn} · Zones: ", style="dim")
        for index, zone in enumerate(self._zones.values()):
            if index:
                line.append(" | ", style="dim")
            count = occupancy.get(zone.name, 0)
            if isinstance(zone, RegZoneSpec):
                capacity = str(zone.max_drones)
            else:
                capacity = "∞"
            line.append(zone.name, style=self._zone_color(zone))
            line.append(f" {count}/{capacity}", style="dim")

        line.append("    Links: ", style="dim")
        connections = {
            connection.identifier(): connection
            for neighbors in simulation.network.adjacency.values()
            for connection in neighbors
        }
        for index, connection in enumerate(connections.values()):
            if index:
                line.append(" | ", style="dim")
            line.append(connection.identifier(), style="cyan")
            line.append(
                f" {connection.current_traffic}/"
                f"{connection.max_link_capacity}",
                style="dim",
            )
        return line

    @classmethod
    def _zone_color(cls, zone: ZoneSpec) -> str:
        """Return a valid configured color or a type-based fallback.

        Args:
            zone (ZoneSpec): Zone whose explicit color or semantic type sets
                the style.

        Returns:
            str: Color string accepted by Rich.
        """
        if zone.color:
            try:
                Color.parse(zone.color)
                return zone.color
            except ColorParseError:
                pass
        return cls._TYPE_COLORS[zone.zone_type]
