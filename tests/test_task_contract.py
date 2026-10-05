"""Additional parser, turn mechanics, and real CLI acceptance requirements."""

from itertools import permutations
from pathlib import Path
import subprocess
import sys

import pytest

from schedule_replay import replay
from test_simulation_regressions import Script
from fly_in.errors import ConfigParseError
from fly_in.parser import Parser
from fly_in.simulation import Simulation


HEADER = "nb_drones: 2\nstart_hub: s 0 0\nend_hub: g 3 0\n"
MOVEMENT_MAP = HEADER + """hub: a 1 0
hub: r 1 1 [zone=restricted]
hub: x 2 0 [zone=blocked]
connection: s-a
connection: a-g
connection: s-r
connection: r-g
connection: s-x
connection: x-g
"""


@pytest.mark.parametrize("text,line,cause", [
    ("", 1, "empty"),
    ("# comment\n\n", 1, "empty"),
    ("\n# comment\nnb_drones: 1.5", 3, "positive_integer"),
    ("nb_drones: +1", 1, "positive_integer"),
    ("nb_drones: 1_000", 1, "positive_integer"),
    ("nb_drones: 1 extra", 1, "positive_integer"),
    (HEADER + "nb_drones: 3", 4, "expected start_hub"),
    (HEADER + "hub: name with_space 1 2", 4, "metadata"),
    (HEADER + "hub: a 1_000 2", 4, "coordinates"),
    (HEADER + "hub: a 1e3 2", 4, "coordinates"),
    (HEADER + "hub: a 1 2 [color=]", 4, "key=value"),
    (HEADER + "hub: a 1 2 [zone=Normal]", 4, "zone_type"),
    (HEADER + "hub: a 1 2 [=red]", 4, "key=value"),
    (HEADER + "connection: s-g-extra", 4, "zone1"),
    (HEADER + "connection: s-", 4, "zone1"),
    (HEADER + "connection: s-g [unknown=1]", 4, "unknown metadata"),
    (HEADER + "connection: s-g [max_link_capacity=1 "
     "max_link_capacity=2]", 4, "more than once"),
])
def test_remaining_parser_errors_report_physical_line_and_cause(
    text: str, line: int, cause: str,
) -> None:
    """Malformed text produces a controlled, actionable parser diagnostic."""
    with pytest.raises(ConfigParseError) as error:
        Parser().parse_text(text)
    assert f"line {line}" in str(error.value)
    assert cause in str(error.value)


@pytest.mark.parametrize("capacity", ["-1", "+1", "1.0", "1_0", "nan"])
def test_link_capacity_requires_positive_integer_text(capacity: str) -> None:
    """Connections must reject coercible or nonnumeric capacity spellings."""
    with pytest.raises(ConfigParseError, match="line 4.*positive integer"):
        Parser().parse_text(
            HEADER + f"connection: s-g [max_link_capacity={capacity}]",
        )


@pytest.mark.parametrize("items", list(permutations(
    ("zone=restricted", "color=arbitrary_colour", "max_drones=2"),
)))
def test_all_metadata_orders_preserve_values(items: tuple[str, ...]) -> None:
    """The specification permits all six orders of the three hub tags."""
    config = Parser().parse_text(
        HEADER + f"hub: a -20 +30 [{' '.join(items)}]",
    )
    zone = config.zones[0]
    assert (zone.x, zone.y, zone.max_drones, zone.color) == (
        -20, 30, 2, "arbitrary_colour",
    )
    assert zone.zone_type.value == "restricted"


def test_unicode_punctuation_names_and_large_fleet_are_preserved() -> None:
    """Names and coordinates need no ASCII identifier or fleet-size cap."""
    config = Parser().parse_text("""# leading comment
nb_drones: 1000000 # any positive fleet size can be parsed
start_hub: база[0] -100 +200 [color=необычный]
end_hub: goal:! 0 0
connection: goal:!-база[0]
""")
    assert config.nb_drones == 1000000
    assert config.start.name == "база[0]"
    assert config.start.color == "необычный"
    assert config.end.name == "goal:!"


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_unreadable_map_path_is_controlled(kind: str, tmp_path: Path) -> None:
    """Both missing files and directories become configuration errors."""
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()
    with pytest.raises(ConfigParseError, match="cannot read"):
        Parser().parse(path)


def snapshot(simulation: Simulation) -> tuple[object, ...]:
    """Capture mutable simulation data before rejecting an illegal row."""
    return (
        simulation.turn, [list(row) for row in simulation.history],
        simulation.metrics.summary(),
        [(drone.current_zone.name, drone.in_transit,
          drone.transit_remaining, list(drone.move_log), drone.delivered)
         for drone in simulation.drones],
        [link.current_traffic for link in
         simulation.network.connections.values()],
    )


@pytest.mark.parametrize("action,cause", [
    ("D1", "invalid movement"),
    ("D0-a", "invalid movement"),
    ("D1-unknown", "unknown"),
    ("D1-g", "no connection"),
    ("D1-r", "movement cost"),
    ("D1-a-s", "movement cost"),
    ("D1-x", "blocked"),
])
def test_illegal_solver_action_has_no_partial_simulation_effects(
    action: str, cause: str,
) -> None:
    """Reject bad identifiers, nonedges, blocked entries and bad durations."""
    simulation = Simulation.from_config(Parser().parse_text(MOVEMENT_MAP),
                                        Script([[action]]))
    before = snapshot(simulation)
    with pytest.raises(ValueError, match=cause):
        simulation.step()
    assert snapshot(simulation) == before


@pytest.mark.parametrize("rows", [
    [["D1-r-s"], ["D1-g"]],
    [["D1-a"], ["D1-g"], ["D1-a"]],
], ids=["in-transit", "already-delivered"])
def test_unavailable_drone_cannot_be_commanded(rows: list[list[str]]) -> None:
    """Neither forced arrivals nor delivered drones accept new departures."""
    simulation = Simulation.from_config(Parser().parse_text(MOVEMENT_MAP),
                                        Script([list(row) for row in rows]))
    for _ in range(len(rows) - 1):
        simulation.step()
    before = snapshot(simulation)
    with pytest.raises(ValueError, match="unavailable drone"):
        simulation.step()
    assert snapshot(simulation) == before


def test_opposing_moves_are_legal_when_shared_link_capacity_is_two() -> None:
    """Hub vacating permits a simultaneous swap when link capacity allows."""
    config = Parser().parse_text(HEADER + """hub: a 1 0
hub: b 2 0
connection: s-a
connection: s-b
connection: a-b [max_link_capacity=2]
connection: b-g
""")
    simulation = Simulation.from_config(config, Script([
        ["D1-a", "D2-b"], ["D1-b", "D2-a"],
        ["D1-g", "D2-b"], ["D2-g"],
    ]))
    result = simulation.run(max_turns=4)
    assert result.turns == 4
    assert replay(config, result.history).cost == 7


@pytest.mark.parametrize("failure", ["syntax", "unreachable", "missing"])
def test_cli_process_reports_errors_without_traceback_or_movements(
    failure: str, tmp_path: Path,
) -> None:
    """Validate error handling through the real process entry point."""
    path = tmp_path / "map.txt"
    if failure != "missing":
        path.write_text("not a map" if failure == "syntax" else HEADER,
                        encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "fly_in", str(path), "--display", "moves",
         "--output-dir", str(tmp_path / "out"),
         "--records-dir", str(tmp_path / "records")],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "error:" in completed.stderr and "Traceback" not in completed.stderr
