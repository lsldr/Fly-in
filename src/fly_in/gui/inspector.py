"""Pinned topology details with live simulation occupancy."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDockWidget, QFormLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from fly_in.models import MapConfig
from fly_in.simulation import Simulation


class InspectorDockWidget(QDockWidget):
    """Show details for one selected hub or undirected connection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create a movable, closable inspector with selectable text."""
        super().__init__("Inspector", parent)
        self.setObjectName("inspectorDock")
        self.setMinimumWidth(290)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        content = QWidget(self)
        content.setStyleSheet("QLabel { font-size: 16px; }")
        layout = QVBoxLayout(content)
        self.placeholder = QLabel("Select a hub or connection to inspect.")
        self.placeholder.setWordWrap(True)
        layout.addWidget(self.placeholder)
        self.form = QFormLayout()
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.addLayout(self.form)
        layout.addStretch()
        self.setWidget(content)
        self.config: MapConfig | None = None
        self.selection: tuple[str, str] | None = None
        self.fields: dict[str, QLabel] = {}
        self._simulation: Simulation | None = None

    def set_config(self, config: MapConfig | None) -> None:
        """Replace topology and clear any pinned selection."""
        self.config = config
        self.selection = None
        self._simulation = None
        self.refresh(None)

    def select_object(self, kind: str, name: str) -> None:
        """Pin a hub or connection using its stable topology identifier."""
        self.selection = (kind, name)
        self.refresh(self._simulation)

    def _field(self, title: str, value: str) -> QLabel:
        """Add a selectable value without interpreting map text as HTML."""
        label = QLabel(value)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        label.setWordWrap(True)
        self.form.addRow(title, label)
        self.fields[title] = label
        return label

    def refresh(self, simulation: Simulation | None) -> None:
        """Refresh the pinned object's live post-turn state."""
        self._simulation = simulation
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields.clear()
        self.placeholder.show()
        if self.config is None or self.selection is None:
            return
        kind, name = self.selection
        if kind in ("hub", "zone"):
            self._hub(name, simulation)
        elif kind in ("link", "connection"):
            self._link(name, simulation)
        self.placeholder.setVisible(not bool(self.fields))

    def _hub(self, name: str, simulation: Simulation | None) -> None:
        """Render hub properties and post-turn occupancy."""
        assert self.config is not None
        config = self.config
        spec = next((zone for zone in
                     [config.start, *config.zones, config.end]
                     if zone.name == name), None)
        if spec is None:
            return
        endpoint = name in (config.start.name, config.end.name)
        occupants = [] if simulation is None else [
            drone.id for drone in simulation.drones
            if not drone.delivered and drone.in_transit is None
            and drone.current_zone.name == name
        ]
        self._field("Name", name)
        self._field("Coordinates", f"({spec.x}, {spec.y})")
        role = ("Start" if name == config.start.name else
                "Goal" if name == config.end.name else "Hub")
        kind = spec.zone_type.value.title()
        if kind == "Restricted":
            kind += " (2 turns)"
        badge = self._field("Type", f"{role} · {kind}")
        badge.setStyleSheet(
            "background: #24283b; border: 1px solid #3b4261;"
            "border-radius: 5px; padding: 4px;"
        )
        capacity = getattr(spec, "max_drones", 1)
        self._field("Capacity", "Unlimited" if endpoint else str(capacity))
        color = spec.color or "Default"
        self._color_field(color)
        self._field("Occupancy", str(len(occupants)))
        self._field("Drones", ", ".join(occupants) or "None")
        self._field("Status", "Full" if not endpoint
                    and len(occupants) >= capacity else "Available")

    def _color_field(self, color: str) -> None:
        """Show a circular swatch beside an always-readable color name."""
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        value = QColor(color)
        fill = value.name() if value.isValid() else "#9aa5ce"
        swatch.setStyleSheet(
            f"background: {fill}; border: 1px solid #9aa5ce;"
            "border-radius: 7px;"
        )
        text = QLabel(color)
        text.setTextFormat(Qt.TextFormat.PlainText)
        text.setWordWrap(True)
        row.addWidget(swatch)
        row.addWidget(text, 1)
        self.form.addRow("Color", holder)
        self.fields["Color"] = text

    def _link(self, name: str, simulation: Simulation | None) -> None:
        """Distinguish traffic during the turn from drones still in transit."""
        assert self.config is not None
        route = next((route for route in self.config.connections
                      if "-".join(sorted((route.from_hub, route.to_hub)))
                      == name or route.name == name), None)
        if route is None:
            return
        identifier = "-".join(sorted((route.from_hub, route.to_hub)))
        traffic = 0
        transit: list[str] = []
        if simulation is not None:
            connection = simulation.network.connections.get(identifier)
            if connection is not None:
                traffic = connection.current_traffic
            transit = [drone.id for drone in simulation.drones
                       if drone.in_transit is not None
                       and drone.in_transit.identifier() == identifier]
        self._field("Connection", identifier)
        self._field("Endpoints", f"{route.from_hub} ↔ {route.to_hub}")
        self._field("Capacity", str(route.max_link_capacity))
        self._field("Traffic this turn", str(traffic))
        self._field("In transit", ", ".join(transit) or "None")
