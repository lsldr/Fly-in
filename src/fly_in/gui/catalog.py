from dataclasses import dataclass
import os
from pathlib import Path

from fly_in.models import MapConfig
from fly_in.errors import ConfigParseError
from fly_in.parser import Parser

IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__", "build", "dist"}


@dataclass(frozen=True)
class MapFile:
    """A discovered map file and its validation result."""

    path: Path
    relative_path: str
    config: MapConfig | None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Return whether this catalog entry parsed successfully."""
        return self.config is not None


class MapCatalog:
    """Discover and validate map files below a selected directory."""

    def __init__(self, parser: Parser | None = None) -> None:
        """Create a catalog using the supplied or default parser."""
        self._parser = parser or Parser()

    def scan(self, root: str | Path) -> list[MapFile]:
        """Discover UTF-8 text maps recursively and retain parsing errors."""
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise NotADirectoryError(f"Map folder does not exist: {root_path}")

        entries: list[MapFile] = []
        for directory, directories, filenames in os.walk(root_path):
            directories[:] = sorted(
                name
                for name in directories
                if not name.startswith(".") and name not in IGNORED_DIRECTORIES
            )
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.suffix.lower() != ".txt":
                    continue
                relative_path = path.relative_to(root_path).as_posix()
                try:
                    config = self._parser.parse(path)
                    entries.append(MapFile(path, relative_path, config))
                except (ConfigParseError, OSError) as error:
                    entries.append(
                        MapFile(path, relative_path, None, str(error))
                    )
        return sorted(entries, key=lambda entry: entry.relative_path)
