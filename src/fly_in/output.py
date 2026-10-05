"""Atomic, specification-formatted simulation output files."""

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fly_in.solvers.protocol import Schedule


class AtomicFileWriter:
    """Own one target and preserve its prior contents until replacement."""

    def __init__(self, path: Path) -> None:
        """Bind the target whose temporary files share its filesystem."""
        self.path = path

    def write(self, content: str) -> None:
        """Replace the UTF-8 target after writing its complete new contents."""
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class OutputWriter:
    """Write one movement file per source map, preserving categories."""

    def __init__(self, output_root: str | Path = "out") -> None:
        """Create a writer rooted at the requested directory."""
        self.output_root = Path(output_root)

    def write(
        self,
        map_path: str | Path,
        history: Schedule,
        map_root: str | Path = "maps",
    ) -> Path:
        """Atomically overwrite movement output after successful execution."""
        source = Path(map_path).resolve()
        try:
            relative = source.relative_to(Path(map_root).resolve())
        except ValueError:
            digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            relative = Path("external") / digest / source.name
        output_path = self.output_root / relative
        content = "\n".join(" ".join(turn) for turn in history)
        if history:
            content += "\n"
        AtomicFileWriter(output_path).write(content)
        return output_path
