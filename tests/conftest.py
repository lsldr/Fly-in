"""Shared offscreen Qt application for GUI requirement checks."""

import os
from typing import Iterator

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture
def app() -> Iterator[QApplication]:
    """Keep a shared Qt application alive for signal and window tests.

    If you have not set QT_QPA_PLATFORM, select the offscreen platform before
    creating the application. Reuse an existing QApplication when available,
    and process queued events when the fixture is released.

    Yields:
        QApplication: Application instance shared with the test's widgets.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    application = (existing if isinstance(existing, QApplication)
                   else QApplication([]))
    yield application
    application.processEvents()
