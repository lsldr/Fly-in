"""Version 2 process, desktop, display, and persistence acceptance checks."""

import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from fly_in.gui.main_window import MainWindow
from fly_in.gui.map_view import MapGraphicsView
from fly_in.gui.planning import PlanningWorker
from fly_in.parser import Parser
from fly_in.solver_factory import SOLVER_CHOICES
from schedule_replay import replay


ROOT = Path(__file__).resolve().parents[1]
UNSOLVABLE = sorted((ROOT / "maps" / "unsolvable").glob("0[1-5]*.txt"))


@pytest.mark.parametrize("solver", SOLVER_CHOICES)
@pytest.mark.parametrize("path", UNSOLVABLE, ids=lambda path: path.stem)
def test_all_cli_solvers_report_unsolvable_without_traceback(
    solver: str, path: Path, tmp_path: Path,
) -> None:
    """A syntactically valid unreachable map fails clearly before movement."""
    completed = subprocess.run(
        [sys.executable, "-m", "fly_in", str(path), "--solver", solver,
         "--display", "moves", "--output-dir", str(tmp_path / "out"),
         "--records-dir", str(tmp_path / "records")],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr
    assert ("unreachable" in completed.stderr or
            "no accessible route" in completed.stderr), completed.stderr
    records = list((tmp_path / "records").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "failed"
    assert not list((tmp_path / "out").rglob("*.txt"))


@pytest.mark.parametrize("solver", SOLVER_CHOICES)
def test_desktop_unsolvable_state_recovers_after_selecting_valid_map(
    solver: str, app: QApplication, tmp_path: Path,
) -> None:
    """Every selected strategy has a controlled failure and can load again."""
    path = tmp_path / "bad.txt"
    path.write_text("nb_drones: 2\nstart_hub: s 0 0\nend_hub: g 1 0\n")
    with patch.object(PlanningWorker, "start", return_value=None):
        window = MainWindow(
            tmp_path, initial_map=path, solver_name=solver, animate=False,
            output_dir=tmp_path / "out", records_dir=tmp_path / "records",
        )
        try:
            window._workers[-1].run()
            app.processEvents()
            # Greedy performs reachability on its first live planning turn.
            if window._ready:
                window._step_simulation()
            assert window._stopped
            assert not window.step_button.isEnabled()
            assert window.reset_button.isEnabled()
            assert window.simulation is not None
            assert window.simulation.turn == 0
            valid = path.read_text() + "connection: s-g\n"
            path.write_text(valid)
            window._load_config(Parser().parse(path), path, valid.encode())
            window._workers[-1].run()
            app.processEvents()
            assert window._ready
            window._step_simulation()
            window._step_simulation()
            assert all(drone.delivered for drone in window.simulation.drones)
        finally:
            window.close()


@pytest.mark.parametrize("display", ["moves", "compact", "full", "visual"])
def test_literal_names_and_unknown_colors_survive_cli_displays(
    display: str, tmp_path: Path,
) -> None:
    """Valid punctuation stays literal and each mode persists legal rows."""
    path = tmp_path / "map.txt"
    path.write_text("""nb_drones: 2
start_hub: <s> 0 0 [color=unlisted_color]
end_hub: [/bold] 2 0 [zone=restricted color=chartreuse]
hub: [red] 1 0 [color=another_color]
connection: <s>-[red]
connection: [red]-[/bold]
""", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "fly_in", str(path), "--display", display,
         "--map-root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
         "--records-dir", str(tmp_path / "records")],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    rows = (tmp_path / "out" / path.name).read_text().splitlines()
    assert len(rows) == 4
    checked = replay(Parser().parse(path), [row.split() for row in rows])
    assert checked.cost == 6
    assert "[/bold]" in completed.stdout
    if display == "moves":
        assert completed.stdout.splitlines() == rows


@pytest.mark.parametrize("exponent", [18, 160, 400])
def test_gui_displays_unbounded_integer_coordinate_spans(
    exponent: int, app: QApplication,
) -> None:
    """Legal integer coordinates must produce finite, usable Qt geometry."""
    extent = 10 ** exponent
    config = Parser().parse_text(
        f"nb_drones: 1\nstart_hub: s {-extent} {extent}\n"
        f"end_hub: g {extent} {-extent}\n"
        "hub: middle 0 0\nconnection: s-middle\nconnection: middle-g\n",
    )
    view = MapGraphicsView()
    try:
        view.resize(600, 400)
        view.show_map(config)
        view.show()
        app.processEvents()
        points = view._zone_positions
        assert points["s"].x() == points["s"].y() == 0
        assert 0 < points["middle"].x() < points["g"].x()
        assert points["middle"].x() * 2 == points["g"].x()
        for point in points.values():
            assert isfinite(point.x()) and isfinite(point.y())
            assert point.x() >= 0
        assert isfinite(view.scene().sceneRect().width())
        assert view.transform().m11() > 0
    finally:
        view.close()
