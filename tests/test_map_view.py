from PySide6.QtGui import QColor

from fly_in.gui.map_view import MapGraphicsView
from fly_in.models import EdgeZoneSpec, MapConfig, RegZoneSpec


def test_zone_colors_use_configured_colors_for_every_zone_kind() -> None:
    config = MapConfig(
        nb_drones=1,
        start=EdgeZoneSpec(name="start", x=0, y=0, color="green"),
        end=EdgeZoneSpec(name="end", x=2, y=0, color="gold"),
        zones=[
            RegZoneSpec(name="hub", x=1, y=0, color="purple"),
        ],
        connections=[],
    )

    for zone in [config.start, *config.zones, config.end]:
        fill, _, _ = MapGraphicsView._zone_colors(zone, config)
        assert QColor(fill) == QColor(zone.color)


def test_normalized_coordinates_and_stable_slots() -> None:
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    from fly_in.realm.entities import Drone
    from fly_in.realm.network import DroneNetwork

    app = QApplication.instance() or QApplication([])
    assert app is not None
    config = MapConfig(
        nb_drones=25,
        start=EdgeZoneSpec(name="s", x=-100, y=100),
        end=EdgeZoneSpec(name="g", x=-98, y=99),
        zones=[], connections=[],
    )
    network = DroneNetwork.from_config(config)
    drones = [Drone(f"D{i}", network.start) for i in range(1, 26)]
    view = MapGraphicsView()
    view.show_map(config)
    assert view._zone_positions["s"] == QPointF(0, 0)
    assert view._zone_positions["g"] == QPointF(260, 130)
    original = view.drone_positions(drones)
    assert len(set((p.x(), p.y()) for p in original.values())) == 25
    for index, first in enumerate(original.values()):
        for last in list(original.values())[index + 1:]:
            assert ((first.x() - last.x()) ** 2
                    + (first.y() - last.y()) ** 2) ** .5 >= 60
    view.show_drones(drones)
    marker = view.drone_items["D2"]
    drones[0].current_zone = network.end
    drones[0].delivered = True
    view.show_drones(drones)
    assert "D1" not in view.drone_items
    assert view.drone_items["D2"] is marker
    assert marker.pos() == original["D2"]
    assert "D1" in view.drone_positions(drones, include_delivered=True)
    view.clear_map()
    assert not view.drone_items
    assert not view.scene().items()


def test_transit_slots_use_midpoint_and_escaped_tooltips() -> None:
    from PySide6.QtWidgets import QApplication

    from fly_in.models import RouteSpec
    from fly_in.realm.entities import Drone
    from fly_in.realm.network import DroneNetwork

    app = QApplication.instance() or QApplication([])
    assert app is not None
    config = MapConfig(
        nb_drones=1,
        start=EdgeZoneSpec(name="<s>", x=0, y=0),
        end=EdgeZoneSpec(name="g", x=2, y=0), zones=[],
        connections=[RouteSpec(from_hub="<s>", to_hub="g")],
    )
    network = DroneNetwork.from_config(config)
    drone = Drone("D1", network.start,
                  network.connection_between(network.start, network.end), 1)
    view = MapGraphicsView()
    view.show_map(config)
    position = view.drone_positions([drone])["D1"]
    assert position.x() == 130
    assert position.y() == -76
    assert "&lt;s&gt;" in view.hub_items["<s>"].toolTip()
    assert "<s>" not in view.hub_items["<s>"].toolTip()


def test_large_coordinates_resize_fit_and_manual_zoom() -> None:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    config = MapConfig(
        nb_drones=1,
        start=EdgeZoneSpec(name="s", x=-(10 ** 12), y=10 ** 12),
        end=EdgeZoneSpec(name="g", x=10 ** 12, y=-(10 ** 12)),
        zones=[], connections=[],
    )
    view = MapGraphicsView()
    view.resize(400, 300)
    view.show_map(config)
    view.show()
    app.processEvents()
    original_scale = view.transform().m11()
    view.resize(800, 600)
    app.processEvents()
    assert view.transform().m11() > original_scale
    assert view._zone_positions["s"] == QPointF(0, 0)
    assert view._zone_positions["g"] == QPointF(260 * 10 ** 12,
                                                260 * 10 ** 12)
    event = QWheelEvent(
        QPointF(200, 100), QPointF(200, 100), QPoint(), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    view.wheelEvent(event)
    manual_scale = view.transform().m11()
    view.resize(600, 400)
    app.processEvents()
    assert view.transform().m11() == manual_scale
    assert not view._fit_mode
    view.fit_map()
    assert view._fit_mode
    view.close()
