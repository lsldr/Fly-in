"""Offscreen rendering and accessibility checks for vector items."""

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene

from fly_in.gui.items import HubItem, QuadcopterItem
from fly_in.models import ZoneType


def test_quadcopter_heading_keeps_item_and_badge_upright() -> None:
    """Render a rotated drone and verify its item transform stays upright."""
    app = QApplication.instance() or QApplication([])
    assert app is not None
    scene = QGraphicsScene()
    drone = QuadcopterItem("D123")
    scene.addItem(drone)
    drone.set_heading(90)
    assert drone.heading == 90
    assert drone.rotation() == 0
    image = QImage(100, 100, QImage.Format.Format_ARGB32)
    image.fill(QColor("#16161e"))
    painter = QPainter(image)
    scene.render(painter)
    painter.end()
    assert not image.isNull()


def test_hub_text_contrast_and_selection_halo() -> None:
    """Check selected hubs show a halo and choose dark ink on light fill."""
    app = QApplication.instance() or QApplication([])
    assert app is not None
    scene = QGraphicsScene()
    hub = HubItem("s", QPointF(), "#808000", "#7dcfff", "start",
                  ZoneType.NORMAL)
    scene.addItem(hub)
    hub.setSelected(True)
    image = QImage(100, 100, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    scene.render(painter)
    painter.end()
    assert hub.halo.isVisible()
    assert hub._ink == QColor("black")


def test_palette_text_contrast_meets_aa() -> None:
    """Check normal-size palette text meets the 4.5:1 contrast threshold."""
    def luminance(hex_color: str) -> float:
        """Convert an sRGB hex color to relative luminance.

        Args:
            hex_color (str): Color accepted by QColor.

        Returns:
            float: Relative luminance between zero and one.
        """
        color = QColor(hex_color)
        channels = [color.redF(), color.greenF(), color.blueF()]
        linear = [value / 12.92 if value <= .04045
                  else ((value + .055) / 1.055) ** 2.4
                  for value in channels]
        return sum(value * weight for value, weight
                   in zip(linear, (.2126, .7152, .0722)))

    for foreground in ("#f8f8f2", "#9aa5ce", "#7dcfff"):
        for background in ("#16161e", "#1a1b26", "#1f2335", "#24283b"):
            ratio = ((luminance(foreground) + .05)
                     / (luminance(background) + .05))
            assert ratio >= 4.5, (foreground, background, ratio)


def test_primary_button_contrast() -> None:
    """Check foreground contrast in normal and active button states."""
    def luminance(hex_color: str) -> float:
        """Convert an sRGB hex color to relative luminance.

        Args:
            hex_color (str): Color accepted by QColor.

        Returns:
            float: Relative luminance between zero and one.
        """
        color = QColor(hex_color)
        channels = [color.redF(), color.greenF(), color.blueF()]
        linear = [value / 12.92 if value <= .04045
                  else ((value + .055) / 1.055) ** 2.4
                  for value in channels]
        return sum(value * weight for value, weight
                   in zip(linear, (.2126, .7152, .0722)))

    for background in ("#7dcfff", "#a6e0ff", "#e0af68", "#f0c788"):
        assert ((luminance(background) + .05)
                / (luminance("#16161e") + .05)) >= 4.5
