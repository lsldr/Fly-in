from pathlib import Path

import pytest

from fly_in.models import MapConfig, ZoneType
from fly_in.errors import ConfigParseError
from fly_in.parser import Parser

EXAMPLE_CONFIG = """\
nb_drones: 5
start_hub: hub 0 0 [color=green]
end_hub: goal 10 10 [color=yellow]
hub: roof1 3 4 [zone=restricted color=red]
hub: roof2 6 2 [zone=normal color=blue]
hub: corridorA 4 3 [zone=priority color=green max_drones=2]
hub: tunnelB 7 4 [zone=normal color=red]
hub: obstacleX 5 5 [zone=blocked color=gray]
connection: hub-roof1
connection: hub-corridorA
connection: roof1-roof2
connection: roof2-goal
connection: corridorA-tunnelB [max_link_capacity=2]
connection: tunnelB-goal
"""


def test_parses_example_into_simulation_config() -> None:
    config = Parser().parse_text(EXAMPLE_CONFIG)

    assert isinstance(config, MapConfig)
    assert config.nb_drones == 5
    assert config.start.name == "hub"
    assert config.start.color == "green"
    assert config.start.x == 0
    assert config.end.name == "goal"
    assert config.end.color == "yellow"
    assert config.zones[0].zone_type is ZoneType.RESTRICTED
    assert config.zones[2].max_drones == 2
    assert config.zones[-1].zone_type is ZoneType.BLOCKED
    assert config.connections[0].name == "hub-roof1"
    assert config.connections[4].max_link_capacity == 2


def test_applies_defaults_and_ignores_endpoint_capacity() -> None:
    config = Parser().parse_text("""\
nb_drones: 1
start_hub: start 0 0 [max_drones=not-a-number]
end_hub: end 1 1 [max_drones=0]
hub: middle 0 1
connection: start-middle
connection: middle-end
""")

    assert config.start.color is None
    assert config.start.zone_type is ZoneType.NORMAL
    assert config.end.color is None
    assert config.zones[0].max_drones == 1
    assert config.connections[0].max_link_capacity == 1


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("hub: room 1.5 2", "coordinates must be integers"),
        ("hub: room 1 2 [zone=unknown]", "zone_type"),
        ("hub: room 1 2 [max_drones=0]", "max_drones"),
        ("hub: room 1 2 [color=red extra=value]", "unknown metadata"),
    ],
)
def test_reports_line_and_cause_for_invalid_zone(
    line: str, message: str
) -> None:
    text = f"nb_drones: 1\nstart_hub: start 0 0\nend_hub: end 1 1\n{line}\n"

    with pytest.raises(ConfigParseError, match=r"line 4") as error:
        Parser().parse_text(text)
    assert message in str(error.value)


def test_connection_must_reference_previously_defined_zones() -> None:
    text = """\
nb_drones: 1
start_hub: start 0 0
connection: start-end
end_hub: end 1 1
"""

    with pytest.raises(ConfigParseError, match=r"line 3.*not-yet-defined"):
        Parser().parse_text(text)


def test_reverse_connection_is_duplicate() -> None:
    text = """\
nb_drones: 1
start_hub: start 0 0
end_hub: end 1 1
connection: start-end
connection: end-start
"""

    with pytest.raises(
        ConfigParseError, match=r"line 5.*duplicate connection"
    ):
        Parser().parse_text(text)


def test_metadata_must_be_well_formed() -> None:
    text = """\
nb_drones: 1
start_hub: start 0 0
end_hub: end 1 1
hub: middle 0 1 [color red]
"""

    with pytest.raises(ConfigParseError, match=r"line 4.*expected key=value"):
        Parser().parse_text(text)


def test_invalid_utf8_becomes_configuration_error(tmp_path: "Path") -> None:
    """Binary map files produce a readable parser error."""
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigParseError, match="cannot read"):
        Parser().parse(path)


@pytest.mark.parametrize("capacity", ["1.0", "-1", "abc"])
def test_zone_capacity_requires_integer_text(capacity: str) -> None:
    """Pydantic coercion must not accept decimal capacity spellings."""
    text = (
        "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 2 0\n"
        f"hub: a 1 0 [max_drones={capacity}]\n"
    )
    with pytest.raises(ConfigParseError, match="positive integer"):
        Parser().parse_text(text)


def test_brackets_in_names_are_not_metadata() -> None:
    """Only fields after the primary tokens form a metadata block."""
    config = Parser().parse_text(
        "nb_drones: 1\nstart_hub: x[y] 0 0 [color=blue]\n"
        "end_hub: end] 1 0\nconnection: x[y]-end] [max_link_capacity=2]\n"
    )
    assert config.start.name == "x[y]"
    assert config.connections[0].max_link_capacity == 2


def test_large_capacity_conversion_keeps_line_context() -> None:
    """Interpreter numeric limits produce configuration errors, not leaks."""
    import sys

    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if not limit:
        pytest.skip("interpreter has no configured integer text limit")
    value = "9" * (limit + 1)
    text = (
        "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 2 0\n"
        f"hub: a 1 0 [max_drones={value}]\n"
    )
    with pytest.raises(ConfigParseError, match="line 4.*max_drones"):
        Parser().parse_text(text)


@pytest.mark.parametrize("prefix", ["start_hub", "end_hub", "hub"])
@pytest.mark.parametrize("metadata", [
    "[color=red][zone=blocked]",
    "[color=red] [zone=blocked]",
    "[color=[red]]",
    "[color=red]]",
    "[[color=red]",
    "[color=red",
    "color=red]",
    "[color=red] trailing",
])
def test_rejects_malformed_zone_metadata_blocks(
    prefix: str, metadata: str,
) -> None:
    """Bracket errors must report the physical line, including comments."""
    lines = ["# comment", "nb_drones: 1", "", "# another comment"]
    lines.append(f"{prefix}: bad 1 0 {metadata}")
    if prefix != "start_hub":
        lines.append("start_hub: s 0 0")
    if prefix != "end_hub":
        lines.append("end_hub: g 2 0")
    with pytest.raises(ConfigParseError, match=r"line 5:.*metadata"):
        Parser().parse_text("\n".join(lines))


@pytest.mark.parametrize("metadata", [
    "[max_link_capacity=1][max_link_capacity=2]",
    "[max_link_capacity=1] [max_link_capacity=2]",
    "[[max_link_capacity=1]]",
    "[max_link_capacity=1] trailing",
])
def test_rejects_malformed_connection_metadata_blocks(metadata: str) -> None:
    """Connections enforce the same block grammar and line context."""
    text = (
        "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\n"
        f"connection: s-g {metadata}\n"
    )
    with pytest.raises(ConfigParseError, match=r"line 4:"):
        Parser().parse_text(text)


@pytest.mark.parametrize("metadata", [
    "zone=restricted color=custom_colour max_drones=2",
    "max_drones=2 zone=restricted color=custom_colour",
    "color=custom_colour max_drones=2 zone=restricted",
])
def test_valid_metadata_order_and_comment_remain_supported(
    metadata: str,
) -> None:
    """Stricter brackets preserve free-form colors, ordering, and names."""
    config = Parser().parse_text(
        "nb_drones: 2\nstart_hub: s 0 0\nend_hub: g 2 0\n"
        f"hub: a[b] -1 +2 [{metadata}] # trailing comment\n"
        "connection: s-a[b]\nconnection: a[b]-g\n"
    )
    zone = config.zones[0]
    assert zone.name == "a[b]"
    assert (zone.x, zone.y) == (-1, 2)
    assert zone.color == "custom_colour"
    assert zone.zone_type is ZoneType.RESTRICTED
    assert zone.max_drones == 2


@pytest.mark.parametrize("text,line,cause", [
    ("nb_drones: 0", 1, "positive integer"),
    ("nb_drones: -2", 1, "positive_integer"),
    ("start_hub: s 0 0", 1, "nb_drones"),
    ("nb_drones: 1\nstart_hub: s 0 0", 2, "missing required end_hub"),
    ("nb_drones: 1\nend_hub: g 1 0", 2, "missing required start_hub"),
    ("nb_drones: 1\nstart_hub: s 0 0\nstart_hub: other 1 0",
     3, "start_hub is defined more than once"),
    ("nb_drones: 1\nend_hub: g 0 0\nend_hub: other 1 0",
     3, "end_hub is defined more than once"),
    ("nb_drones: 1\nstart_hub: s 0 0\nhub: s 1 0",
     3, "already defined"),
    ("nb_drones: 1\nstart_hub: s 0 0\nhub: bad-name 1 0",
     3, "cannot contain dashes"),
    ("nb_drones: 1\nstart_hub: s 0 0\nhub: a 1 0 [color=red color=blue]",
     3, "specified more than once"),
    ("nb_drones: 1\nstart_hub: s 0 0\nhub: a 1 0 [max_link_capacity=2]",
     3, "not valid for a zone"),
    ("nb_drones: 1\nstart_hub: s 0 0\nconnection: s-s",
     3, "different zones"),
    ("nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\n"
     "connection: s-g [max_link_capacity=0]", 4, "max_link_capacity"),
    ("nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\n"
     "connection: s-g [color=red]", 4, "not valid for a connection"),
])
def test_parser_requirement_errors_keep_line_and_cause(
    text: str, line: int, cause: str,
) -> None:
    """Cover first records, endpoint uniqueness, names, and field scopes."""
    with pytest.raises(ConfigParseError) as error:
        Parser().parse_text(text)
    assert f"line {line}" in str(error.value)
    assert cause in str(error.value)
