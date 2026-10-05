"""Selectable topology items and persistent vector drone markers."""

from math import cos, pi, sin

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsPathItem, QStyleOptionGraphicsItem, QWidget,
)

from fly_in.models import ZoneType


class HubItem(QGraphicsPathItem):
    """Hub shape with a separate, persistent selection halo."""

    def __init__(
        self, name: str, position: QPointF, fill: str, outline: str,
        role: str, zone_type: ZoneType,
    ) -> None:
        """Create a role-specific shape and its selection halo."""
        super().__init__()
        path = QPainterPath()
        if role in {"start", "goal"}:
            sides = 6 if role == "start" else 4
            for index in range(sides):
                angle = 2 * pi * index / sides - pi / 2
                point = QPointF(17 * cos(angle), 17 * sin(angle))
                if index:
                    path.lineTo(point)
                else:
                    path.moveTo(point)
            path.closeSubpath()
        else:
            path.addEllipse(QRectF(-17, -17, 34, 34))
        if zone_type is ZoneType.RESTRICTED:
            path.addEllipse(QRectF(-22, -22, 44, 44))
        self.setPath(path)
        self.setPos(position)
        self.setBrush(QColor(fill))
        self.setPen(QPen(QColor(outline), 2.5))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setData(0, "zone")
        self.setData(1, name)
        self.setZValue(3)
        self.role = role
        self.zone_type = zone_type
        color = QColor(fill)
        channels = [color.redF(), color.greenF(), color.blueF()]
        linear = [value / 12.92 if value <= .04045
                  else ((value + .055) / 1.055) ** 2.4
                  for value in channels]
        luminance = sum(channel * weight for channel, weight
                        in zip(linear, (.2126, .7152, .0722)))
        self._ink = QColor("black" if luminance > .179 else "white")
        halo = QPainterPath()
        halo.addEllipse(QRectF(-27, -27, 54, 54))
        self.halo = QGraphicsPathItem(halo, self)
        self.halo.setPen(QPen(QColor("#7dcfff"), 3,
                              Qt.PenStyle.DashLine))
        self.halo.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.halo.hide()

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint redundant role/type symbols and selection feedback."""
        super().paint(painter, option, widget)
        self.halo.setVisible(self.isSelected())
        painter.setPen(QPen(self._ink, 2))
        painter.setFont(QFont("Sans Serif", 11, QFont.Weight.Bold))
        marker = {"start": "S", "goal": "G"}.get(self.role, "")
        if not marker:
            marker = {ZoneType.PRIORITY: "◆",
                      ZoneType.BLOCKED: "×"}.get(self.zone_type, "")
        painter.drawText(QRectF(-16, -16, 32, 32),
                         Qt.AlignmentFlag.AlignCenter, marker)


class LinkItem(QGraphicsPathItem):
    """Wide clickable link with a thin visible centerline."""

    def __init__(self, name: str, first: QPointF, last: QPointF) -> None:
        """Create a selectable path with a generous hit area."""
        path = QPainterPath(first)
        path.lineTo(last)
        super().__init__(path)
        self.setPen(QPen(QColor(0, 0, 0, 1), 16))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setData(0, "connection")
        self.setData(1, name)
        self.setZValue(0)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the visible line and optional dashed selection."""
        if self.isSelected():
            painter.setPen(QPen(
                QColor("#7dcfff"), 7, Qt.PenStyle.DashLine
            ))
            painter.drawPath(self.path())
        painter.setPen(QPen(QColor("#9aa5ce"), 2.2))
        painter.drawPath(self.path())


class QuadcopterItem(QGraphicsItem):
    """Vector quadcopter whose heading never rotates its ID badge."""

    def __init__(self, drone_id: str) -> None:
        """Create a persistent drone marker with an upright badge."""
        super().__init__()
        self.drone_id = drone_id
        self.heading = 0.0
        self.setZValue(7)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        """Include the arms, rotor rings, and identifier badge."""
        return QRectF(-19, -19, 38, 46)

    def set_heading(self, degrees: float) -> None:
        """Rotate only the body; the ID remains upright."""
        self.heading = degrees
        self.update()

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Draw four rotors and an upright, high-contrast ID badge."""
        painter.save()
        painter.rotate(self.heading)
        painter.setPen(QPen(QColor("#7dcfff"), 2))
        for x, y in ((-10, -10), (10, -10), (-10, 10), (10, 10)):
            painter.drawLine(QPointF(), QPointF(x, y))
            painter.drawEllipse(QPointF(x, y), 3.5, 3.5)
        painter.setBrush(QColor("#2ac3de"))
        painter.setPen(QPen(QColor("#16161e"), 1.5))
        painter.drawEllipse(QPointF(), 7, 7)
        painter.drawLine(QPointF(0, 0), QPointF(7, 0))
        painter.restore()
        painter.setPen(QColor("#16161e"))
        painter.setBrush(QColor("#f8f8f2"))
        painter.drawRoundedRect(QRectF(-17, 13, 34, 13), 3, 3)
        painter.setFont(QFont("Sans Serif", 7, QFont.Weight.Bold))
        painter.drawText(QRectF(-17, 13, 34, 13),
                         Qt.AlignmentFlag.AlignCenter, self.drone_id)
