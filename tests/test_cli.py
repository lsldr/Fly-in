"""Regression coverage for command selection and entrypoint behavior."""

from pathlib import Path
from unittest.mock import patch

import pytest

from fly_in.cli import CliApplication, main
from fly_in.solver_factory import SolverFactory


MAP = """nb_drones: 1
start_hub: start 0 0
end_hub: goal 1 0
connection: start-goal
"""


def test_select_nested_maps_rejects_zero_and_negative(tmp_path: Path) -> None:
    """Menu indices are one-based even when maps are nested."""
    folder = tmp_path / "easy"
    folder.mkdir()
    first = folder / "a.txt"
    first.write_text(MAP)
    (tmp_path / "z.map").write_text(MAP)
    with patch("builtins.input", side_effect=["0", "-1", "3", "1"]):
        assert CliApplication().select_map(tmp_path) == first


def test_noninteractive_missing_map_is_usage_error() -> None:
    """Piped input does not silently enter an interactive prompt."""
    with patch("sys.stdin.isatty", return_value=False):
        with pytest.raises(SystemExit) as error:
            main([])
    assert error.value.code == 2


def test_validation_skips_gui(tmp_path: Path) -> None:
    """Validation takes precedence over the rendering mode."""
    path = tmp_path / "map.txt"
    path.write_text(MAP)
    with patch("fly_in.cli.CliApplication.launch_gui") as launch:
        main([str(path), "--validate", "--gui"])
    launch.assert_not_called()


def test_gui_receives_selected_solver_and_directories(tmp_path: Path) -> None:
    """The CLI forwards configuration without importing Qt in terminal mode."""
    path = tmp_path / "map.txt"
    path.write_text(MAP)
    with patch("fly_in.cli.CliApplication.launch_gui") as launch:
        main([str(path), "--gui", "--solver", "greedy",
              "--map-root", str(tmp_path), "--output-dir", str(tmp_path),
              "--records-dir", str(tmp_path)])
    launch.assert_called_once_with(
        path, tmp_path, "greedy", tmp_path, tmp_path,
    )


def test_interactive_map_then_mode(tmp_path: Path) -> None:
    """No-argument use prompts for both a map and playback mode."""
    path = tmp_path / "map.txt"
    path.write_text(MAP)
    with patch("sys.stdin.isatty", return_value=True):
        with patch("builtins.input", side_effect=["1", "2"]):
            with patch("fly_in.cli.CliApplication.launch_gui") as launch:
                main(["--map-root", str(tmp_path)])
    assert launch.call_args.args[0] == path


def test_unknown_solver_rejected() -> None:
    """Programmatic callers cannot accidentally get another solver."""
    with pytest.raises(ValueError, match="unknown solver"):
        SolverFactory().create("unknown")


def test_cli_modes_are_mutually_exclusive() -> None:
    """Conflicting playback flags produce argparse usage errors."""
    with pytest.raises(SystemExit) as error:
        main(["--gui", "--terminal"])
    assert error.value.code == 2
