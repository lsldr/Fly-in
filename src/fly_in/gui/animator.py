"""Interruptible interpolation of a single simulation turn."""
from collections.abc import Mapping
from math import atan2, degrees

from PySide6.QtCore import (
    QAbstractAnimation, QEasingCurve, QObject, QPointF, QVariantAnimation,
    Signal,
)
from PySide6.QtWidgets import QGraphicsItem


class TurnAnimationController(QObject):
    """Move persistent graphics items without advancing simulation state."""

    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Create a reusable controller owned by the containing view."""
        super().__init__(parent)
        self.animation = QVariantAnimation(self)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.valueChanged.connect(self._update)
        self.animation.finished.connect(self._finish)
        self._tracks: list[tuple[QGraphicsItem, QPointF, QPointF]] = []

    @property
    def is_running(self) -> bool:
        """Return whether a turn is active, including a paused turn."""
        return self.animation.state() != QAbstractAnimation.State.Stopped

    @property
    def is_paused(self) -> bool:
        """Return whether the active turn is suspended."""
        return self.animation.state() == QAbstractAnimation.State.Paused

    def start(
        self, items: Mapping[str, QGraphicsItem],
        targets: dict[str, QPointF], duration: int, animate: bool = True,
    ) -> None:
        """Capture current positions and animate toward the given targets."""
        self.cancel()
        for name, target in targets.items():
            item = items.get(name)
            if item is None:
                continue
            origin = QPointF(item.pos())
            destination = QPointF(target)
            self._tracks.append((item, origin, destination))
            delta = destination - origin
            heading = getattr(item, "set_heading", None)
            if callable(heading) and not delta.isNull():
                heading(degrees(atan2(delta.y(), delta.x())))
        if not animate or duration <= 0 or not self._tracks:
            self._finish()
            return
        self.animation.setDuration(duration)
        self.animation.start()

    def pause(self) -> None:
        """Pause interpolation at its current positions."""
        if self.animation.state() == QAbstractAnimation.State.Running:
            self.animation.pause()

    def resume(self) -> None:
        """Continue a suspended interpolation."""
        if self.is_paused:
            self.animation.resume()

    def cancel(self) -> None:
        """Discard the turn without delivering a completion signal."""
        self.animation.stop()
        self._tracks.clear()

    def _update(self, value: object) -> None:
        """Apply an eased interpolation fraction to each captured item."""
        if not isinstance(value, (float, int)):
            return
        for item, origin, target in self._tracks:
            item.setPos(origin + (target - origin) * float(value))

    def _finish(self) -> None:
        """Snap to the exact final positions and notify the owner."""
        for item, _, target in self._tracks:
            item.setPos(target)
        self._tracks.clear()
        self.finished.emit()
