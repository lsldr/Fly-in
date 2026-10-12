"""Planning failures must reach Qt's failure path and permit recovery."""

from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from fly_in.gui.main_window import MainWindow
from fly_in.gui.planning import PlanningWorker
from fly_in.parser import Parser


MAP = """nb_drones: 1
start_hub: s 0 0
end_hub: g 1 0
connection: s-g
"""


@pytest.mark.parametrize("obstruction", ["records_file", "log_directory"])
def test_adaptive_filesystem_errors_emit_failure_once(
    tmp_path: Path, app: QApplication, obstruction: str,
) -> None:
    """Report either records-file or log-directory I/O failure exactly once.

    Args:
        tmp_path (Path): Temporary directory containing the test's records
            obstruction.
        app (QApplication): QApplication fixture used to deliver queued worker
            signals.
        obstruction (str): Parametrized filesystem state, ``records_file`` or
            ``log_directory``.
    """
    records = tmp_path / "records"
    if obstruction == "records_file":
        records.write_text("occupied", encoding="utf-8")
    else:
        (records / "run_log.jsonl").mkdir(parents=True)
    worker = PlanningWorker(17, Parser().parse_text(MAP), "adaptive",
                            {"time_budget": 0}, records_dir=records)
    failures: list[tuple[int, str]] = []
    successes: list[int] = []
    worker.failed.connect(lambda generation, message:
                          failures.append((generation, message)))
    worker.prepared.connect(lambda generation, result, elapsed:
                            successes.append(generation))
    worker.run()
    app.processEvents()
    assert len(failures) == 1
    assert failures[0][0] == 17
    assert failures[0][1]
    assert str(records) in failures[0][1]
    assert successes == []


@pytest.mark.parametrize("cancelled", [False, True])
def test_worker_handles_permission_errors_and_suppresses_cancelled_results(
    tmp_path: Path, app: QApplication, cancelled: bool,
) -> None:
    """Report an injected permission error unless cancellation was requested.

    Args:
        tmp_path (Path): Pytest temporary directory used as the records
            location.
        app (QApplication): QApplication fixture used to deliver queued worker
            signals.
        cancelled (bool): Whether cancellation suppresses the worker result.
    """
    worker = PlanningWorker(9, Parser().parse_text(MAP), "adaptive",
                            records_dir=tmp_path)
    failures: list[tuple[int, str]] = []
    successes: list[int] = []
    worker.failed.connect(lambda generation, message:
                          failures.append((generation, message)))
    worker.prepared.connect(lambda generation, result, elapsed:
                            successes.append(generation))
    if cancelled:
        worker.cancel()
    with patch("fly_in.gui.planning.SolverFactory.create",
               side_effect=PermissionError("cannot write adaptive log")):
        worker.run()
    app.processEvents()
    assert failures == ([] if cancelled else
                        [(9, "cannot write adaptive log")])
    assert successes == []


def test_adaptive_gui_stops_and_can_reset_after_logging_error(
    tmp_path: Path, app: QApplication,
) -> None:
    """Exercise worker failure UI, save failure, and recovery after repair.

    Args:
        tmp_path (Path): Pytest temporary directory for map and records files.
        app (QApplication): QApplication fixture used to process worker
            signals.
    """
    path = tmp_path / "simple.txt"
    path.write_text(MAP, encoding="utf-8")
    records = tmp_path / "records"
    records.write_text("occupied", encoding="utf-8")
    # Drive run on the test thread so an escaping exception fails pytest
    # directly, without PySide swallowing it in a background QThread.
    with patch.object(PlanningWorker, "start", return_value=None):
        window = MainWindow(
            tmp_path, initial_map=path, solver_name="adaptive",
            solver_parameters={"time_budget": 0}, records_dir=records,
            output_dir=tmp_path / "out", animate=False,
        )
        try:
            worker = window._workers[-1]
            worker.run()
            app.processEvents()
            assert window._stopped
            assert not window._ready
            assert not window._playing
            assert window.simulation_state.text() == "Simulation stopped"
            assert not window.play_button.isEnabled()
            assert not window.step_button.isEnabled()
            assert window.reset_button.isEnabled()
            assert str(records) in window.statusBar().currentMessage()
            assert not (tmp_path / "out").exists()
            records.unlink()
            records.mkdir()
            window._reset_simulation()
            window._workers[-1].run()
            app.processEvents()
            assert window._ready
            assert window.simulation_state.text() == "Ready to launch"
            assert window.play_button.isEnabled()
            assert window.step_button.isEnabled()
        finally:
            window.close()
