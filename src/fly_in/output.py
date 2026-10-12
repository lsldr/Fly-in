"""Atomic, specification-formatted simulation output files."""

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fly_in.solvers.protocol import Schedule


class AtomicFileWriter:
    """Atomic text-file writer for one target path.

    Attributes:
        path (Path): Destination replaced after its new contents are written.
    """

    def __init__(self, path: Path) -> None:
        """Bind a target whose temporary file will share its filesystem.

        Args:
            path (Path): Destination replaced after complete content is
                written and flushed.
        """
        self.path = path

    def write(self, content: str) -> None:
        """Atomically replace the UTF-8 target with complete new contents.

        Args:
            content (str): Text to write before replacing the destination.

        Raises:
            OSError: If the destination cannot be created, written, flushed,
                or replaced.
        """
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
    """Movement-file writer that preserves source map categories.

    Attributes:
        output_root (Path): Base directory for all movement output files.
    """

    def __init__(self, output_root: str | Path = "out") -> None:
        """Create a writer rooted at the requested directory.

        Args:
            output_root (str | Path): Base directory for movement files.
        """
        self.output_root = Path(output_root)

    def write(
        self,
        map_path: str | Path,
        history: Schedule,
        map_root: str | Path = "maps",
    ) -> Path:
        """Write one map's movement history as an atomic text file.

        Maps inside ``map_root`` keep their relative directory structure.
        External maps are placed under a stable hash directory.

        Args:
            map_path (str | Path): Source map whose output is being written.
            history (Schedule): Movement events grouped by simulation turn.
            map_root (str | Path): Root used to preserve map categories.

        Returns:
            Path: Output file path under ``output_root``.

        Raises:
            OSError: If the output directory or file cannot be written.
        """
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
