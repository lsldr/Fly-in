"""Bounded, interactive map canvas with persistent drone markers."""

from html import escape
from math import atan2, cos, degrees, pi, sin

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPen, QResizeEvent, QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsTextItem, QGraphicsView, QWidget,
)

from fly_in.gui.items import HubItem, LinkItem, QuadcopterItem
from fly_in.models import MapConfig, RegZoneSpec, ZoneSpec, ZoneType
from fly_in.realm.entities import Drone
from fly_in.simulation import Simulation


class MapGraphicsView(QGraphicsView):
    """Interactive scene view for topology and stable drone slots.

    Signals:
        object_selected (Signal[str, str]): Emits object kind and identifier.

    Attributes:
        NODE_RADIUS (int): Base node radius in scene units.
        GRID_STEP (int): Scene distance for one map coordinate unit.
        drone_items (dict[str, QuadcopterItem]): Persistent drone graphics.
        hub_items (dict[str, HubItem]): Hub graphics keyed by zone name.
        link_items (dict[str, LinkItem]): Links keyed by canonical identifier.
    """

    object_selected = Signal(str, str)
    NODE_RADIUS = 17
    GRID_STEP = 130

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the scene, selections, and stable slot cache.

        Args:
            parent (QWidget | None): Optional Qt owner for this graphics view.
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setBackgroundBrush(QColor("#16161e"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.scene().selectionChanged.connect(self._selection_changed)
        self._zone_positions: dict[str, QPointF] = {}
        self.drone_items: dict[str, QuadcopterItem] = {}
        self.hub_items: dict[str, HubItem] = {}
        self.link_items: dict[str, LinkItem] = {}
        self._slots: dict[str, dict[str, int]] = {}
        self._config: MapConfig | None = None
        self._fit_scale = 1.0
        self._fit_mode = True
        self._fitting = False

    def clear_map(self) -> None:
        """Clear all scene items, slot assignments, and map selection."""
        self.drone_items.clear()
        self.hub_items.clear()
        self.link_items.clear()
        self._slots.clear()
        self._zone_positions.clear()
        self._config = None
        self._fit_mode = True
        self.scene().clear()

    def show_map(self, config: MapConfig) -> None:
        """Build the topology and fit its complete item bounds.

        Map coordinates are scaled by ``GRID_STEP`` scene units and the
        vertical axis is inverted so increasing map y points upward.

        Args:
            config (MapConfig): Map topology and visual properties to display.
        """
        self.clear_map()
        self._config = config
        specs = [config.start, *config.zones, config.end]
        minimum_x = min(zone.x for zone in specs)
        maximum_y = max(zone.y for zone in specs)
        positions = {
            zone.name: QPointF(
                (zone.x - minimum_x) * self.GRID_STEP,
                (maximum_y - zone.y) * self.GRID_STEP,
            ) for zone in specs
        }
        self._zone_positions = positions
        for route in config.connections:
            first, last = positions[route.from_hub], positions[route.to_hub]
            name = "-".join(sorted((route.from_hub, route.to_hub)))
            item = LinkItem(name, first, last)
            item.setToolTip(
                f"<b>{escape(name)}</b><br>Capacity: "
                f"{route.max_link_capacity}<br>Traffic: 0"
            )
            self.link_items[name] = item
            self.scene().addItem(item)
            self._label(f"CAP {route.max_link_capacity}",
                        (first + last) / 2 + QPointF(5, 5), "#9aa5ce")
        for zone in specs:
            fill, outline, _ = self._zone_colors(zone, config)
            role = ("start" if zone is config.start else
                    "goal" if zone is config.end else "hub")
            hub = HubItem(zone.name, positions[zone.name], fill,
                          outline, role, zone.zone_type)
            hub.setToolTip(self._zone_tooltip(zone))
            self.hub_items[zone.name] = hub
            self.scene().addItem(hub)
            label = self._label(zone.name, QPointF(), "#f8f8f2")
            label.setPos(positions[zone.name] + QPointF(
                -label.boundingRect().width() / 2, 29
            ))
        self._update_bounds()
        self.fit_map()

    def _label(
        self, text: str, position: QPointF, color: str,
    ) -> QGraphicsTextItem:
        """Create a plain-text topology label at a scene position.

        Args:
            text (str): Label content, rendered without HTML interpretation.
            position (QPointF): Label origin in scene coordinates.
            color (str): Text color accepted by Qt.

        Returns:
            QGraphicsTextItem: The inserted scene text item.
        """
        label = QGraphicsTextItem()
        label.setPlainText(text)
        label.setDefaultTextColor(QColor(color))
        label.setFont(QFont("Sans Serif", 12))
        label.setPos(position)
        label.setZValue(5)
        label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene().addItem(label)
        return label

    @staticmethod
    def _slot_position(index: int) -> QPointF:
        """Place a slot in expanding rings around its location center.

        Args:
            index (int): Zero-based slot assigned to an item at one location.

        Returns:
            QPointF: Offset in scene units from the location center.
        """
        radius = 76.0
        # A 64-pixel chord separates complete markers, including badges.
        count = 6
        while index >= count:
            index -= count
            radius += 64
            count = max(6, int(2 * pi * radius / 64))
        angle = -pi / 2 + 2 * pi * index / count
        return QPointF(radius * cos(angle), radius * sin(angle))

    def drone_positions(
        self, drones: list[Drone], include_delivered: bool = False,
    ) -> dict[str, QPointF]:
        """Allocate stable per-location slots and return scene positions.

        Existing occupants keep their slot when others leave or arrive.
        Delivered positions can be requested for final arrival animation.

        Args:
            drones (list[Drone]): Fleet whose current locations determine slot
                groups.
            include_delivered (bool): Include delivered drones at their current
                zone when preparing the final arrival animation.

        Returns:
            dict[str, QPointF]: Scene coordinates keyed by drone ID. Drones
            whose location has no
            displayed center are omitted.
        """
        groups: dict[str, list[Drone]] = {}
        centers: dict[str, QPointF] = {}
        for drone in drones:
            if drone.delivered and not include_delivered:
                continue
            if drone.in_transit is None:
                key = "zone:" + drone.current_zone.name
                center = self._zone_positions.get(drone.current_zone.name)
            else:
                edge = drone.in_transit
                key = "link:" + edge.identifier()
                first = self._zone_positions.get(edge.zone_a.name)
                last = self._zone_positions.get(edge.zone_b.name)
                center = ((first + last) / 2
                          if first is not None and last is not None else None)
            if center is not None:
                centers[key] = center
                groups.setdefault(key, []).append(drone)
        for key in list(self._slots):
            ids = {drone.id for drone in groups.get(key, [])}
            self._slots[key] = {
                drone_id: slot for drone_id, slot in self._slots[key].items()
                if drone_id in ids
            }
        result: dict[str, QPointF] = {}
        for key, group in groups.items():
            slots = self._slots.setdefault(key, {})
            for drone in sorted(group, key=lambda item: (
                int(item.id[1:]) if item.id[1:].isdigit() else 0, item.id
            )):
                if drone.id not in slots:
                    slot = 0
                    while slot in slots.values():
                        slot += 1
                    slots[drone.id] = slot
                result[drone.id] = (centers[key]
                                    + self._slot_position(slots[drone.id]))
        return result

    def show_drones(self, drones: list[Drone]) -> None:
        """Update marker positions and remove drones no longer displayed.

        Args:
            drones (list[Drone]): Current fleet state; delivered drones are
                omitted.
        """
        positions = self.drone_positions(drones)
        for drone_id in list(self.drone_items):
            if drone_id not in positions:
                self.scene().removeItem(self.drone_items.pop(drone_id))
        for drone in drones:
            if drone.id not in positions:
                continue
            item = self.drone_items.get(drone.id)
            if item is None:
                item = QuadcopterItem(drone.id)
                self.drone_items[drone.id] = item
                self.scene().addItem(item)
            target = positions[drone.id]
            delta = target - item.pos()
            if delta.manhattanLength() > .01:
                item.set_heading(degrees(atan2(delta.y(), delta.x())))
            item.setPos(target)
            location = (drone.in_transit.identifier()
                        if drone.in_transit else drone.current_zone.name)
            item.setToolTip(
                f"<b>{escape(drone.id)}</b><br>Location: {escape(location)}"
                f"<br>Remaining transit turns: {drone.transit_remaining}"
            )
        self._update_bounds()

    def refresh(self, simulation: Simulation) -> None:
        """Refresh live hub occupancy and link traffic tooltips.

        Args:
            simulation (Simulation): Simulation snapshot whose current state is
                displayed.
        """
        if self._config is None:
            return
        for zone in [self._config.start, *self._config.zones,
                     self._config.end]:
            ids = [drone.id for drone in simulation.drones
                   if not drone.delivered and drone.in_transit is None
                   and drone.current_zone.name == zone.name]
            self.hub_items[zone.name].setToolTip(
                self._zone_tooltip(zone) + f"<br>Occupancy: {len(ids)}"
                + "<br>Drones: " + escape(", ".join(ids) or "none")
            )
        for name, item in self.link_items.items():
            edge = simulation.network.connections[name]
            ids = [drone.id for drone in simulation.drones
                   if drone.in_transit is not None
                   and drone.in_transit.identifier() == name]
            item.setToolTip(
                f"<b>{escape(name)}</b><br>Capacity: "
                f"{edge.max_link_capacity}<br>Traffic this turn: "
                f"{edge.current_traffic}<br>In transit: "
                + escape(", ".join(ids) or "none")
            )

    def _update_bounds(self) -> None:
        """Include all current items and proportional scene padding."""
        bounds = self.scene().itemsBoundingRect()
        horizontal = max(60.0, bounds.width() * .2)
        vertical = max(60.0, bounds.height() * .2)
        self.scene().setSceneRect(bounds.adjusted(
            -horizontal, -vertical, horizontal, vertical
        ))

    def fit_map(self) -> None:
        """Fit topology, labels, and drone markers into the viewport."""
        self._fit_mode = True
        if self.scene().items() and not self._fitting:
            self._fitting = True
            try:
                self._update_bounds()
                self.fitInView(self.scene().sceneRect(),
                               Qt.AspectRatioMode.KeepAspectRatio)
                self._fit_scale = self.transform().m11()
            finally:
                self._fitting = False

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Refit the overview after resize while fit mode is enabled.

        Args:
            event (QResizeEvent): Qt resize event delivered for this view.
        """
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_map()

    def drawBackground(self, painter: QPainter, rect: QRect | QRectF) -> None:
        """Draw a dot grid while bounding the number of visible points.

        Args:
            painter (QPainter): Qt painter for the background region.
            rect (QRect | QRectF): Visible scene rectangle in scene
                coordinates.
        """
        super().drawBackground(painter, rect)
        painter.save()
        painter.setPen(QPen(QColor("#24283b"), 1))
        grid_step = 40
        while ((rect.width() / grid_step + 1)
               * (rect.height() / grid_step + 1) > 10000):
            grid_step *= 2
        left = int(rect.left()) // grid_step * grid_step
        top = int(rect.top()) // grid_step * grid_step
        for x in range(left, int(rect.right()) + grid_step, grid_step):
            for y in range(top, int(rect.bottom()) + grid_step, grid_step):
                painter.drawPoint(QPointF(x, y))
        painter.restore()

    def _selection_changed(self) -> None:
        """Emit the first selected item's kind and stable identifier."""
        selected = self.scene().selectedItems()
        if not selected:
            self.object_selected.emit("", "")
        else:
            self.object_selected.emit(str(selected[0].data(0)),
                                      str(selected[0].data(1)))

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom around the pointer within bounds relative to the fitted view.

        Args:
            event (QWheelEvent): Wheel event whose vertical delta determines
                zoom direction.
        """
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        scale = self.transform().m11() * factor / max(self._fit_scale, 1e-300)
        if .25 <= scale <= 16:
            self._fit_mode = False
            self.scale(factor, factor)
        event.accept()

    @staticmethod
    def _zone_colors(
        zone: ZoneSpec, config: MapConfig,
    ) -> tuple[str, str, str]:
        """Resolve configured fill and type-based outline colors.

        Args:
            zone (ZoneSpec): Zone whose color and role determine its fill.
            config (MapConfig): Map endpoints used to identify start and goal.

        Returns:
            tuple[str, str, str]: Fill color, outline color, and descriptive
            label for the zone's role or type.
        """
        requested = QColor(zone.color) if zone.color else QColor()
        if zone is config.start:
            fill, outline, label = "#73daca", "#2ac3de", "Start"
        elif zone is config.end:
            fill, outline, label = "#e0af68", "#ff9e64", "Goal"
        elif zone.zone_type is ZoneType.BLOCKED:
            fill, outline, label = "#565f89", "#9aa5ce", "Blocked zone"
        elif zone.zone_type is ZoneType.RESTRICTED:
            fill, outline, label = "#ff9e64", "#ff9e64", "Restricted zone"
        elif zone.zone_type is ZoneType.PRIORITY:
            fill, outline, label = "#bb9af7", "#bb9af7", "Priority zone"
        else:
            fill, outline, label = "#7aa2f7", "#7dcfff", "Normal zone"
        return (requested.name() if requested.isValid() else fill,
                outline, label)

    @staticmethod
    def _zone_tooltip(zone: ZoneSpec) -> str:
        """Format escaped hub properties for a hover tooltip.

        Args:
            zone (ZoneSpec): Zone whose name, type, coordinates, capacity, and
                color are displayed.

        Returns:
            str: HTML tooltip text with user-provided strings escaped.
        """
        capacity = (str(zone.max_drones) if isinstance(zone, RegZoneSpec)
                    else "unlimited")
        return (f"<b>{escape(zone.name)}</b><br>Type: {zone.zone_type.value}"
                f"<br>Coordinates: ({zone.x}, {zone.y})"
                f"<br>Capacity: {capacity}<br>Color: "
                + escape(zone.color or "default"))
