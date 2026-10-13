"""Parser boundary cases for valid but disconnected maps."""

import sys

import pytest

from fly_in.errors import ConfigParseError
from fly_in.parser import Parser


@pytest.mark.parametrize(
    "records",
    [
        "",
        "hub: island 1 0\nconnection: start-island\n",
        "hub: island 1 0\nhub: other 2 0\n"
        "connection: start-island\nconnection: other-goal\n",
    ],
    ids=["no-links", "one-sided-component", "separate-components"],
)
def test_syntactically_valid_disconnected_maps_parse(records: str) -> None:
    """Leave graph reachability to the solver, not the text parser.

    Args:
        records (str): Valid zone and link records forming a disconnected map.
    """
    text = (
        "nb_drones: 3\n"
        "start_hub: start 0 0\n"
        "end_hub: goal 3 0\n"
        f"{records}"
    )

    config = Parser().parse_text(text)

    assert config.nb_drones == 3
    assert config.start.name == "start"
    assert config.end.name == "goal"


def test_disconnected_map_still_rejects_malformed_record() -> None:
    """Disconnection does not make malformed syntax acceptable."""
    text = """\
nb_drones: 1
start_hub: start 0 0
end_hub: goal 3 0
hub: island 1 0
connection: start-island
hub: invalid 2 0 [max_drones=0]
"""

    with pytest.raises(ConfigParseError, match=r"line 6.*max_drones"):
        Parser().parse_text(text)


@pytest.mark.parametrize(
    "name",
    ["a:b", "éclair", "a[b]", "a/b,;!?=~+"],
    ids=["colon", "unicode", "brackets", "punctuation"],
)
def test_single_token_unicode_and_punctuation_names_round_trip(
    name: str,
) -> None:
    """Accept names with punctuation when they contain no spaces or dashes.

    Args:
        name (str): A one-token name using otherwise valid characters.
    """
    text = (
        "nb_drones: 1\n"
        f"start_hub: {name} 0 0\n"
        "end_hub: goal 1 0\n"
        f"connection: {name}-goal\n"
    )

    config = Parser().parse_text(text)

    assert config.start.name == name
    assert config.connections[0].from_hub == name


@pytest.mark.parametrize("comment", [" # trailing", "#trailing"])
def test_comment_after_closed_metadata_is_ignored(comment: str) -> None:
    """Allow comments immediately after or separated from a metadata block.

    Args:
        comment (str): Comment spelling appended to a valid hub record.
    """
    config = Parser().parse_text(
        "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 2 0\n"
        f"hub: a 1 0 [color=violet,]{comment}\n"
    )

    assert config.zones[0].color == "violet,"


def test_comment_that_removes_metadata_closer_is_a_line_error() -> None:
    """A comment cannot turn an unfinished metadata block into valid syntax."""
    text = (
        "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 2 0\n"
        "hub: a 1 0 [color=violet # comment hides the closing bracket]\n"
    )

    with pytest.raises(ConfigParseError, match=r"line 4.*metadata"):
        Parser().parse_text(text)


@pytest.mark.parametrize(
    "record,line,cause",
    [
        ("header", 1, "positive integer"),
        ("coordinate", 4, "coordinates must be integers"),
        ("link_capacity", 4, "positive integer"),
    ],
)
def test_integer_digit_limit_is_reported_as_parse_error(
    record: str, line: int, cause: str,
) -> None:
    """Convert oversized integer text into a controlled parser diagnostic.

    Args:
        record (str): Integer field whose text exceeds the interpreter limit.
        line (int): Expected physical source line in the error.
        cause (str): Expected diagnostic cause.
    """
    digit_limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if not digit_limit:
        pytest.skip("interpreter has no configured integer text limit")
    oversized = "9" * (digit_limit + 1)
    if record == "header":
        text = f"nb_drones: {oversized}\n"
    elif record == "coordinate":
        text = (
            "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 2 0\n"
            f"hub: a {oversized} 0\n"
        )
    else:
        text = (
            "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 2 0\n"
            "connection: s-g "
            f"[max_link_capacity={oversized}]\n"
        )

    with pytest.raises(ConfigParseError) as error:
        Parser().parse_text(text)
    assert f"line {line}" in str(error.value)
    assert cause in str(error.value)
