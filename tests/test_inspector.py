"""Live inspector occupancy and text safety checks."""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from fly_in.gui.inspector import InspectorDockWidget
from fly_in.models import (
    EdgeZoneSpec, MapConfig, RegZoneSpec, RouteSpec, ZoneType,
)
from fly_in.simulation import Simulation
from fly_in.solvers.prioritized import PriorityBasedSolver


def test_inspector_tracks_restricted_transit_and_arrival() -> None:
    """Check plain-text names, capacities, occupancy, traffic, and arrivals."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    config = MapConfig(
        nb_drones=1,
        start=EdgeZoneSpec(name="<start>", x=0, y=0),
        end=EdgeZoneSpec(name="goal", x=2, y=0),
        zones=[RegZoneSpec(name="r", x=1, y=0,
                           zone_type=ZoneType.RESTRICTED)],
        connections=[RouteSpec(from_hub="<start>", to_hub="r"),
                     RouteSpec(from_hub="r", to_hub="goal")],
    )
    simulation = Simulation.from_config(config, PriorityBasedSolver())
    dock = InspectorDockWidget()
    dock.set_config(config)
    dock.select_object("hub", "<start>")
    dock.refresh(simulation)
    assert dock.fields["Name"].text() == "<start>"
    assert dock.fields["Name"].textFormat() == Qt.TextFormat.PlainText
    assert dock.fields["Capacity"].text() == "Unlimited"
    assert dock.fields["Drones"].text() == "D1"
    simulation.step()
    dock.refresh(simulation)
    assert dock.fields["Occupancy"].text() == "0"
    dock.select_object("connection", "<start>-r")
    assert dock.fields["In transit"].text() == "D1"
    assert dock.fields["Traffic this turn"].text() == "1"
    simulation.step()
    dock.refresh(simulation)
    assert dock.fields["In transit"].text() == "None"
    assert dock.fields["Traffic this turn"].text() == "0"
    dock.select_object("hub", "r")
    assert dock.fields["Status"].text() == "Full"
    simulation.step()
    dock.select_object("hub", "goal")
    dock.refresh(simulation)
    assert dock.fields["Occupancy"].text() == "0"
    dock.set_config(None)
    assert dock.fields == {}
    app.processEvents()
    dock.close()
