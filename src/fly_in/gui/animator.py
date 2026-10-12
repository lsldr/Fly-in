"""Interruptible interpolation of a single simulation turn."""
from collections.abc import Mapping
from math import atan2, degrees

from PySide6.QtCore import (
    QAbstractAnimation, QEasingCurve, QObject, QPointF, QVariantAnimation,
    Signal,
)
from PySide6.QtWidgets import QGraphicsItem


class TurnAnimationController(QObject):
    """Controller for graphics interpolation during a simulation turn.

    Signals:
        finished (Signal): Emitted when tracked items reach their destinations.

    Attributes:
        animation (QVariantAnimation): Reusable animation that drives progress.
    """

    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Create a reusable controller owned by the containing view.

        Args:
            parent (QObject | None): Optional Qt owner that controls this
                controller's lifetime.
        """
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
        """Whether an animation is running or paused.

        Returns:
            bool: True unless the underlying Qt animation is stopped.
        """
        return self.animation.state() != QAbstractAnimation.State.Stopped

    @property
    def is_paused(self) -> bool:
        """Whether the active animation is suspended.

        Returns:
            bool: True when the underlying Qt animation is paused.
        """
        return self.animation.state() == QAbstractAnimation.State.Paused

    def start(
        self, items: Mapping[str, QGraphicsItem],
        targets: dict[str, QPointF], duration: int, animate: bool = True,
    ) -> None:
        """Capture item positions and move them toward their target points.

        Items absent from ``items`` are skipped. If animation is disabled, has
        no tracks, or has a nonpositive duration, this method places all
        tracked items at their targets and emits ``finished`` immediately.

        Args:
            items (Mapping[str, QGraphicsItem]): Graphics items indexed by the
                names used in ``targets``.
            targets (dict[str, QPointF]): Destination scene coordinates keyed
                by item name.
            duration (int): Animation duration in milliseconds.
            animate (bool): Whether to interpolate over ``duration``
                milliseconds.
        """
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
        """Pause an actively running interpolation at its current positions."""
        if self.animation.state() == QAbstractAnimation.State.Running:
            self.animation.pause()

    def resume(self) -> None:
        """Resume interpolation only when the animation is paused."""
        if self.is_paused:
            self.animation.resume()

    def cancel(self) -> None:
        """Stop interpolation and discard tracks without signaling.

        The ``finished`` signal is not emitted when a turn is cancelled.
        """
        self.animation.stop()
        self._tracks.clear()

    def _update(self, value: object) -> None:
        """Apply an eased interpolation fraction to every tracked item.

        Args:
            value (object): Numeric progress from the Qt animation, usually in
                [0, 1].
        """
        if not isinstance(value, (float, int)):
            return
        for item, origin, target in self._tracks:
            item.setPos(origin + (target - origin) * float(value))

    def _finish(self) -> None:
        """Snap tracked items to their destinations and emit ``finished``."""
        for item, _, target in self._tracks:
            item.setPos(target)
        self._tracks.clear()
        self.finished.emit()
