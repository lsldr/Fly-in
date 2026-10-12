"""Desktop session lifecycle and shared output integration."""

import json
import os
from pathlib import Path
from time import monotonic, sleep
from typing import Callable

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fly_in.gui.main_window import MainWindow
from fly_in.solver_factory import DEFAULT_SOLVER
from fly_in.solvers.adaptive import AdaptiveSolver
from fly_in.solvers.greedy import GreedySolver


SIMPLE_MAP = """nb_drones: 1
start_hub: start 0 0
end_hub: goal 2 0
hub: restricted 1 0 [zone=restricted]
connection: start-restricted
connection: restricted-goal
"""


def wait_for(app: QApplication, predicate: Callable[[], bool]) -> None:
    """Process queued events until a condition succeeds or five seconds pass.

    Args:
        app (QApplication): QApplication whose event queue should be processed.
        predicate (Callable[[], bool]): Zero-argument condition checked between
            event iterations.

    Raises:
        AssertionError: If the condition remains false at the deadline.
    """
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        app.processEvents()
        sleep(0.001)
    assert predicate()


def make_window(tmp_path: Path, animate: bool = False) -> MainWindow:
    """Create a window with a simple map and temporary artifact paths.

    Args:
        tmp_path (Path): Pytest temporary directory used for map and outputs.
        animate (bool): Whether turn transitions should use timed
            interpolation.

        Returns:
            MainWindow: Window using temporary map and output paths.
    """
    root = tmp_path / "maps"
    root.mkdir(exist_ok=True)
    path = root / "simple.txt"
    path.write_text(SIMPLE_MAP, encoding="utf-8")
    return MainWindow(
        root, initial_map=path, output_dir=tmp_path / "out",
        records_dir=tmp_path / "records", animate=animate,
    )


def get_app() -> QApplication:
    """Return a QApplication configured for headless rendering.

    Returns:
        QApplication: Existing QApplication or a newly created offscreen
            instance.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    return app if isinstance(app, QApplication) else QApplication([])


def test_loaded_gui_map_is_paused_and_saves_both_outputs(
    tmp_path: Path,
) -> None:
    """Check paused loading, single-step progress, and one-time persistence.

    Args:
        tmp_path (Path): Pytest temporary directory for map and run artifacts.
    """
    app = get_app()
    window = make_window(tmp_path)
    wait_for(app, lambda: window._ready)
    assert window.simulation is not None
    assert window.simulation.turn == 0
    assert not window._playing
    assert window.simulation_state.text() == "Ready to launch"
    for turn in range(1, 4):
        window._step_simulation()
        assert window.simulation.turn == turn
    assert window.progress.value() == 100
    assert window.play_button.text() == "✓ Completed"
    records = list((tmp_path / "records").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "completed"
    assert record["metrics"]["arrival_turns"] == {"D1": 3}
    assert record["metrics"]["total_path_cost"] == 3
    assert record["solver"] == DEFAULT_SOLVER
    assert (tmp_path / "out" / "simple.txt").read_text() == (
        "D1-restricted-start\nD1-restricted\nD1-goal\n"
    )
    window.close()
    assert len(list((tmp_path / "records").glob("*.json"))) == 1


def test_adaptive_worker_preserves_parameters_and_logs_once(
    tmp_path: Path,
) -> None:
    """Check adaptive options and run logging across a reset.

    Args:
        tmp_path (Path): Pytest temporary directory for map and solver records.
    """
    app = get_app()
    root = tmp_path / 'maps'
    root.mkdir()
    path = root / 'simple.txt'
    path.write_text(SIMPLE_MAP, encoding='utf-8')
    options = {'time_budget': 0.02, 'seed': 4}
    window = MainWindow(
        root, initial_map=path, solver_name='adaptive',
        output_dir=tmp_path / 'out', records_dir=tmp_path / 'records',
        solver_parameters=options, animate=False,
    )
    wait_for(app, lambda: window._ready)
    assert window.simulation is not None
    assert isinstance(window.simulation.solver, AdaptiveSolver)
    assert window.simulation.turn == 0
    for _ in range(3):
        window._step_simulation()
    records = list((tmp_path / 'records').glob('*.json'))
    assert len(records) == 1
    assert json.loads(records[0].read_text())['solver_parameters'] == options
    log_path = tmp_path / 'records' / 'run_log.jsonl'
    assert len(log_path.read_text().splitlines()) == 1
    window._reset_simulation()
    wait_for(app, lambda: window._ready)
    assert window.solver_name == 'adaptive'
    assert window.solver_parameters == options
    assert len(log_path.read_text().splitlines()) == 2
    window.close()


def test_speed_slider_right_increases_speed(tmp_path: Path) -> None:
    """Check slider bounds, default duration, and reversed visual direction.

    Args:
        tmp_path (Path): Pytest temporary directory for the fixture map.
    """
    app = get_app()
    window = make_window(tmp_path)
    window.show()
    wait_for(app, lambda: window._ready)
    slider = window.speed_slider
    assert slider.minimum() == 100
    assert slider.maximum() == 2000
    assert slider.value() == 500
    assert slider.invertedAppearance()

    center = slider.height() // 2
    QTest.mouseClick(
        slider, Qt.MouseButton.LeftButton, pos=QPoint(2, center),
    )
    app.processEvents()
    slower_duration = slider.value()
    QTest.mouseClick(
        slider, Qt.MouseButton.LeftButton,
        pos=QPoint(slider.width() - 2, center),
    )
    app.processEvents()
    faster_duration = slider.value()
    assert slower_duration > faster_duration
    assert window.speed_label.text().startswith(
        f'{faster_duration} ms / turn'
    )
    window.close()


def test_pause_resume_and_reset_cancel_animation(tmp_path: Path) -> None:
    """Check pause/resume state and rejection of callbacks after reset.

    Args:
        tmp_path (Path): Pytest temporary directory for map and artifacts.
    """
    app = get_app()
    window = make_window(tmp_path, animate=True)
    wait_for(app, lambda: window._ready)
    window._toggle_playback()
    assert window.animator.is_running
    window._toggle_playback()
    assert window.animator.is_paused
    assert not window._playing
    window._toggle_playback()
    assert not window.animator.is_paused
    generation = window._generation
    window._reset_simulation()
    assert not window.animator.is_running
    window._next_turn(generation)
    wait_for(app, lambda: window._ready)
    assert window.simulation is not None
    assert window.simulation.turn == 0
    window.close()


def test_reset_preserves_solver_and_loaded_source(tmp_path: Path) -> None:
    """Check reset uses loaded source bytes after the map file is deleted.

    Args:
        tmp_path (Path): Pytest temporary directory containing the source map.
    """
    app = get_app()
    window = make_window(tmp_path)
    wait_for(app, lambda: window._ready)
    window.solver_name = "greedy"
    path = window.loaded_map_path
    assert path is not None
    path.unlink()
    window._reset_simulation()
    wait_for(app, lambda: window._ready)
    assert window.simulation is not None
    assert isinstance(window.simulation.solver, GreedySolver)
    assert window.simulation.turn == 0
    window.close()


def test_reload_clears_session_and_stale_results(tmp_path: Path) -> None:
    """Check reload clears state and rejects a result from the old generation.

    Args:
        tmp_path (Path): Pytest temporary directory containing the map library.
    """
    app = get_app()
    window = make_window(tmp_path)
    wait_for(app, lambda: window._ready)
    previous = window.simulation
    generation = window._generation
    window.reload_maps()
    window._prepared(generation, previous, 0.1)
    assert window.simulation is None
    assert window.loaded_config is None
    assert not window._ready
    assert not window.play_button.isEnabled()
    window.close()


def test_map_replacement_during_planning(tmp_path: Path) -> None:
    """Check that replacing a planning job leaves only the latest run ready.

    Args:
        tmp_path (Path): Pytest temporary directory containing the source map.
    """
    app = get_app()
    window = make_window(tmp_path)
    first_generation = window._generation
    window._reset_simulation()
    wait_for(app, lambda: window._ready)
    assert window._generation > first_generation
    assert window.simulation is not None
    assert window.simulation.turn == 0
    window.close()


def test_failed_planning_leaves_reset_available(tmp_path: Path) -> None:
    """Check unreachable topology records failure while leaving reset enabled.

    Args:
        tmp_path (Path): Pytest temporary directory for map and run artifacts.
    """
    app = get_app()
    root = tmp_path / "maps"
    root.mkdir()
    path = root / "bad.txt"
    path.write_text("nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\n")
    window = MainWindow(
        root, initial_map=path, records_dir=tmp_path / "records",
        output_dir=tmp_path / "out", animate=False,
    )
    wait_for(app, lambda: window._stopped)
    assert not window.play_button.isEnabled()
    assert window.reset_button.isEnabled()
    records = list((tmp_path / "records").glob("*.json"))
    assert json.loads(records[0].read_text())["status"] == "failed"
    assert not (tmp_path / "out").exists()
    window.close()


def test_paused_turn_inspection_and_single_step(tmp_path: Path) -> None:
    """Check inspection follows a completed turn after stepping from pause.

    Args:
        tmp_path (Path): Pytest temporary directory for map and run artifacts.
    """
    app = get_app()
    window = make_window(tmp_path, animate=True)
    wait_for(app, lambda: window._ready)
    window._toggle_playback()
    window._toggle_playback()
    assert window.animator.is_paused
    window._object_selected("zone", "start")
    assert window.inspector.fields["Occupancy"].text() == "1"
    window._step_simulation()
    assert not window.animator.is_paused
    assert not window._playing
    window.animator.animation.setCurrentTime(window.speed_slider.value())
    app.processEvents()
    assert window.simulation is not None
    assert window.simulation.turn == 1
    assert window.inspector.fields["Occupancy"].text() == "0"
    assert not window.animator.is_running
    window._reset_simulation()
    wait_for(app, lambda: window._ready)
    assert window.play_button.property("playing") is False
    window.close()
