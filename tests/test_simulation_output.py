"""Movement, rendering, and file-output integration coverage."""

from pathlib import Path

import pytest

from fly_in.cli import CliApplication
from fly_in.models import MapConfig
from fly_in.output import OutputWriter
from fly_in.parser import Parser
from fly_in.realm.entities import Drone
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solvers.protocol import Schedule
from fly_in.visualization.terminal import TerminalVisualizer


class MixedMovementSolver:
    """Issue one ordinary and one restricted movement together."""

    def __init__(self) -> None:
        """Initialize a solver that emits its scripted row only once."""
        self.planned = False

    def plan(
        self, network: DroneNetwork, drones: list[Drone]
    ) -> Schedule:
        """Return mixed normal and restricted actions on the first call.

        Args:
            network (DroneNetwork): Network being simulated.
            drones (list[Drone]): Current fleet; unused by this fixed script.

        Returns:
            Schedule: One action row on the first call, then an empty row.
        """
        if self.planned:
            return [[]]
        self.planned = True
        return [["D1-normal", "D2-passage-start"]]


MIXED_MOVEMENT_MAP = """\
nb_drones: 2
start_hub: start 0 0
end_hub: goal 3 0
hub: normal 1 0
hub: passage 1 1 [zone=restricted color=orange]
connection: start-normal
connection: normal-goal
connection: start-passage
connection: passage-goal
"""


def test_mixed_normal_and_restricted_moves_advance_correctly() -> None:
    """Advance an ordinary move alongside a restricted-link departure."""
    config = Parser().parse_text(MIXED_MOVEMENT_MAP)
    simulation = Simulation.from_config(config, MixedMovementSolver())

    first_turn = simulation.step()

    assert first_turn == ["D1-normal", "D2-passage-start"]
    assert simulation.drones[0].current_zone.name == "normal"
    assert simulation.drones[0].in_transit is None
    assert simulation.drones[1].current_zone.name == "start"
    assert simulation.drones[1].in_transit is not None

    second_turn = simulation.step()

    assert second_turn == ["D2-passage"]
    assert simulation.drones[1].current_zone.name == "passage"
    assert simulation.drones[1].in_transit is None


def test_output_writer_formats_turns_and_overwrites_in_category_tree(
    tmp_path: Path,
) -> None:
    """Format turn rows and replace output under the source category.

    Args:
        tmp_path (Path): Isolated filesystem root for maps and output.
    """
    map_root = tmp_path / "maps"
    map_path = map_root / "easy" / "linear.txt"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("unused map contents", encoding="utf-8")
    writer = OutputWriter(tmp_path / "out")
    history = [["D1-room", "D2-corridorA"], ["D1-goal"]]

    output_path = writer.write(map_path, history, map_root)
    assert output_path.read_text(encoding="utf-8") == (
        "D1-room D2-corridorA\nD1-goal\n"
    )
    writer.write(map_path, [["D1-goal"]], map_root)

    assert output_path == tmp_path / "out" / "easy" / "linear.txt"
    assert output_path.read_text(encoding="utf-8") == "D1-goal\n"


def test_terminal_renderer_shows_colored_moves_and_zone_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Render movement, zone state, and links to terminal output.

    Args:
        capsys (pytest.CaptureFixture[str]): Captured stdout and stderr.
    """
    config: MapConfig = Parser().parse_text(MIXED_MOVEMENT_MAP)
    simulation = Simulation.from_config(config, MixedMovementSolver())
    moves = simulation.step()

    TerminalVisualizer(config, display="full").render_turn(simulation, moves)
    output = capsys.readouterr().out

    assert "D1-normal D2-passage-start" in output
    assert "Zones:" in output
    assert "Links:" in output


def test_map_picker_lists_nested_category_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """List category-relative paths and select the indexed map.

    Args:
        tmp_path (Path): Temporary map-library root.
        monkeypatch (pytest.MonkeyPatch): Test input substitution fixture.
        capsys (pytest.CaptureFixture[str]): Captured prompt and menu output.
    """
    maps = tmp_path / "maps"
    (maps / "easy").mkdir(parents=True)
    (maps / "challenger").mkdir()
    (maps / "easy" / "linear.txt").write_text("map", encoding="utf-8")
    (maps / "challenger" / "dream.txt").write_text(
        "map", encoding="utf-8"
    )
    monkeypatch.setattr("builtins.input", lambda: "2")

    selected = CliApplication().select_map(maps)

    assert selected == maps / "easy" / "linear.txt"
    assert "1. challenger/dream.txt" in capsys.readouterr().err
