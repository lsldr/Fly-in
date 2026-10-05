"""Deterministic interpolation and interruption checks."""
import os

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QGraphicsEllipseItem

from fly_in.gui.animator import TurnAnimationController


def test_interpolation_pause_cancel_and_synchronous_completion() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    item = QGraphicsEllipseItem()
    controller = TurnAnimationController()
    finished: list[bool] = []
    controller.finished.connect(lambda: finished.append(True))
    controller.start({"D1": item}, {"D1": QPointF(100, 40)}, 1000)
    controller.animation.setCurrentTime(500)
    assert item.pos() == QPointF(50, 20)
    controller.pause()
    assert controller.is_running and controller.is_paused
    controller.resume()
    assert controller.is_running and not controller.is_paused
    controller.cancel()
    app.processEvents()
    assert not controller.is_running
    assert finished == []
    assert item.pos() == QPointF(50, 20)
    controller.start({"D1": item}, {"D1": QPointF(100, 40)}, 1000,
                     animate=False)
    assert item.pos() == QPointF(100, 40)
    assert finished == [True]


def test_completion_sets_exact_target() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    item = QGraphicsEllipseItem()
    controller = TurnAnimationController()
    controller.start({"D1": item}, {"D1": QPointF(-10, 70)}, 100)
    controller.animation.setCurrentTime(100)
    assert item.pos() == QPointF(-10, 70)
    assert not controller.is_running
    app.processEvents()
