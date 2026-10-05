"""Small terminal network schemes with counts at hubs and on transit links."""

from collections import Counter

from rich.color import Color, ColorParseError
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from fly_in.models import MapConfig, RegZoneSpec, RouteSpec, ZoneSpec, ZoneType
from fly_in.simulation import Simulation


class TerminalScene:
    """Render exact counts, falling back to edge rows when geometry crowds."""

    COLORS = {
        ZoneType.NORMAL: "bright_blue",
        ZoneType.PRIORITY: "bright_magenta",
        ZoneType.RESTRICTED: "yellow",
        ZoneType.BLOCKED: "bright_black",
    }

    def __init__(self, config: MapConfig) -> None:
        """Assign stable IDs and widths that do not change between turns."""
        self.config = config
        self.zones: list[ZoneSpec] = [
            config.start, *config.zones, config.end,
        ]
        self.indices = {
            zone.name: index for index, zone in enumerate(self.zones, 1)
        }
        self.by_name = {zone.name: zone for zone in self.zones}
        self.id_digits = max(2, len(str(len(self.zones))))
        self.count_digits = len(str(config.nb_drones))
        self._legend_shown = False

    @classmethod
    def color(cls, zone: ZoneSpec) -> str:
        """Honor supported map colors with a safe type-based fallback."""
        if zone.color:
            try:
                Color.parse(zone.color)
                return zone.color
            except ColorParseError:
                pass
        return cls.COLORS[zone.zone_type]

    def _identifier(self, zone: ZoneSpec) -> str:
        """Encode endpoint or zone type without depending on color."""
        if zone is self.config.start:
            role = "S"
        elif zone is self.config.end:
            role = "G"
        else:
            role = {ZoneType.NORMAL: "H", ZoneType.PRIORITY: "P",
                    ZoneType.RESTRICTED: "R", ZoneType.BLOCKED: "X"}[
                        zone.zone_type]
        return f"{role}{self.indices[zone.name]:0{self.id_digits}d}"

    def _hub(self, name: str, occupancy: Counter[str]) -> str:
        """Format a hub with active occupancy, excluding delivered drones."""
        identifier = self._identifier(self.by_name[name])
        return f"[{identifier}:{occupancy[name]:{self.count_digits}d}]"

    def _travelers(
        self, route: RouteSpec, transit: Counter[tuple[str, str]],
        reserve: bool = False,
    ) -> str:
        """Format transit counts by destination, independent of traffic."""
        edge = "-".join(sorted((route.from_hub, route.to_hub)))
        parts = []
        for name in (route.from_hub, route.to_hub):
            zone = self.by_name[name]
            if zone.zone_type is not ZoneType.RESTRICTED:
                continue
            count = (self.config.nb_drones if reserve
                     else transit[edge, name])
            if count:
                parts.append(f"{count}->{self._identifier(zone)}")
        return "{" + ",".join(parts) + "}" if parts else ""

    def render(
        self, simulation: Simulation, moves: list[str], width: int,
    ) -> Group:
        """Render a post-turn scheme; show the name/capacity legend once."""
        occupancy: Counter[str] = Counter()
        transit: Counter[tuple[str, str]] = Counter()
        delivered = 0
        for drone in simulation.drones:
            if drone.delivered:
                delivered += 1
            elif drone.in_transit is not None:
                edge = drone.in_transit
                transit[edge.identifier(),
                        edge.other_end(drone.current_zone).name] += 1
            else:
                occupancy[drone.current_zone.name] += 1
        heading = Text(
            f"Turn {simulation.turn} | At hubs {sum(occupancy.values())}"
            f" | In transit {sum(transit.values())}"
            f" | Delivered {delivered}/{len(simulation.drones)}",
            style="bold cyan",
        )
        spatial = self._spatial(occupancy, transit, width)
        scheme = spatial if spatial is not None else self._edge_rows(
            occupancy, transit
        )
        title = "Network" if spatial is not None else "Network · connections"
        hint = ("Crossings are not hubs" if spatial is not None else
                "Repeated IDs refer to the same hub; counts are not summed")
        panel = Panel(scheme, title=title, subtitle=hint, padding=(0, 1))
        if not self._legend_shown:
            self._legend_shown = True
            return Group(self._legend(), heading, panel, Text())
        return Group(heading, panel, Text())

    def _legend(self) -> Text:
        """Explain compact tokens and retain exact names and capacities."""
        legend = Text(
            "[hub:count] = active drones; {count->hub} = in transit, "
            "arriving next turn.\n"
            "S start · G goal · H normal · P priority · R restricted · "
            "X blocked\nGoal count excludes delivered drones. "
            "Every connection is bidirectional.\n",
        )
        for index, zone in enumerate(self.zones):
            if index:
                legend.append(" · ")
            capacity = (str(zone.max_drones)
                        if isinstance(zone, RegZoneSpec) else "unlimited")
            legend.append(
                f"{self._identifier(zone)}={zone.name} "
                f"({zone.zone_type.value}, cap {capacity})",
                style=self.color(zone),
            )
        legend.append("\n")
        return legend

    def _edge_rows(
        self, occupancy: Counter[str], transit: Counter[tuple[str, str]],
    ) -> Text:
        """Render every link exactly once, plus any isolated hubs."""
        output = Text()
        connected: set[str] = set()
        for route in self.config.connections:
            connected.update((route.from_hub, route.to_hub))
            output.append(self._hub(route.from_hub, occupancy),
                          style=self.color(self.by_name[route.from_hub]))
            travelers = self._travelers(route, transit)
            output.append(f" --{travelers}-- " if travelers else " <--> ",
                          style="bold yellow" if travelers else "cyan")
            output.append(self._hub(route.to_hub, occupancy),
                          style=self.color(self.by_name[route.to_hub]))
            output.append("\n")
        for zone in self.zones:
            if zone.name not in connected:
                output.append(
                    self._hub(zone.name, occupancy) + " (isolated)\n",
                    style=self.color(zone),
                )
        return output

    def _spatial(
        self, occupancy: Counter[str], transit: Counter[tuple[str, str]],
        width: int,
    ) -> Text | None:
        """Use coordinates only when all count labels fit without collisions.

        Space for every possible restricted-transit label is reserved even at
        Turn 0, keeping the layout independent of changing drone counts.
        """
        columns = min(120, width - 6)
        token_width = len(self._hub(self.zones[0].name, occupancy))
        if columns < token_width:
            return None
        min_x, max_x = (min(z.x for z in self.zones),
                        max(z.x for z in self.zones))
        min_y, max_y = (min(z.y for z in self.zones),
                        max(z.y for z in self.zones))
        positions = {
            zone.name: (
                round((zone.x-min_x) / max(1, max_x-min_x)
                      * (columns-token_width)),
                round((max_y-zone.y) / max(1, max_y-min_y) * 12),
            ) for zone in self.zones
        }
        occupied: dict[tuple[int, int], str] = {}
        for name, (x, y) in positions.items():
            for offset in range(token_width):
                if (x+offset, y) in occupied:
                    return None
                occupied[x+offset, y] = name
        rows = max(y for _, y in positions.values()) + 1
        cells = [[" " for _ in range(columns)] for _ in range(rows)]
        styles: dict[tuple[int, int], str] = {}
        paths: list[set[tuple[int, int]]] = []
        labels: list[tuple[RouteSpec, int, int, int]] = []
        reserved: dict[tuple[int, int], int] = {}
        for index, route in enumerate(self.config.connections):
            x1, y1 = positions[route.from_hub]
            x2, y2 = positions[route.to_hub]
            x1 += token_width // 2
            x2 += token_width // 2
            steps = max(abs(x2-x1), abs(y2-y1), 1)
            path = {(round(x1+(x2-x1)*step/steps),
                     round(y1+(y2-y1)*step/steps))
                    for step in range(steps+1)}
            endpoints = {route.from_hub, route.to_hub}
            for point in path:
                if point in occupied and occupied[point] not in endpoints:
                    return None
            paths.append(path)
            size = len(self._travelers(route, transit, reserve=True))
            if size:
                x, y = round((x1+x2-size)/2), round((y1+y2)/2)
                if x < 0 or x+size > columns:
                    return None
                for offset in range(size):
                    point = x+offset, y
                    if point in occupied or point in reserved:
                        return None
                    reserved[point] = index
                labels.append((route, x, y, size))
        for index, path in enumerate(paths):
            route = self.config.connections[index]
            style = ("bold yellow" if self._travelers(route, transit)
                     else "cyan")
            for x, y in path:
                if (x, y) in reserved and reserved[x, y] != index:
                    return None
                cells[y][x] = "."
                styles[x, y] = style
        for name, (x, y) in positions.items():
            for offset, char in enumerate(self._hub(name, occupancy)):
                cells[y][x+offset] = char
                styles[x+offset, y] = "bold " + self.color(self.by_name[name])
        for route, x, y, size in labels:
            label = self._travelers(route, transit)
            if label:
                for offset, char in enumerate(label.center(size)):
                    cells[y][x+offset] = char
                    styles[x+offset, y] = "bold yellow"
        result = Text()
        for y, row in enumerate(cells):
            for x, char in enumerate(row):
                result.append(char, style=styles.get((x, y), ""))
            if y != rows-1:
                result.append("\n")
        return result
