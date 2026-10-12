"""Recursive map discovery and parse-error retention checks."""

from pathlib import Path

from fly_in.gui.catalog import MapCatalog

VALID_MAP = """\
nb_drones: 1
start_hub: start 0 0
end_hub: end 1 1
connection: start-end
"""


def test_catalog_finds_valid_and_invalid_maps_recursively(
    tmp_path: Path,
) -> None:
    """Discover nested maps, retain invalid entries, and skip hidden folders.

    Args:
        tmp_path (Path): Pytest temporary directory containing the map tree.
    """
    maps = tmp_path / "maps"
    nested = maps / "challenge"
    nested.mkdir(parents=True)
    (maps / "linear.txt").write_text(VALID_MAP)
    (nested / "broken.txt").write_text("nb_drones: zero\n")
    ignored = tmp_path / ".venv" / "maps"
    ignored.mkdir(parents=True)
    (ignored / "ignored.txt").write_text(VALID_MAP)

    entries = MapCatalog().scan(tmp_path)

    assert [entry.relative_path for entry in entries] == [
        "maps/challenge/broken.txt",
        "maps/linear.txt",
    ]
    assert entries[0].config is None
    assert "line 1" in (entries[0].error or "")
    assert entries[1].is_valid
