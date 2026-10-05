"""Shared offscreen Qt application for GUI requirement checks."""

import os
from typing import Iterator

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture
def app() -> Iterator[QApplication]:
    """Keep Qt alive while exercising connected signals and windows."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    application = (existing if isinstance(existing, QApplication)
                   else QApplication([]))
    yield application
    application.processEvents()
