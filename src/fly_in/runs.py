"""Shared run provenance and finalization for terminal and GUI clients."""

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fly_in.output import AtomicFileWriter, OutputWriter
from fly_in.simulation import Simulation
from fly_in.provenance import PlanningProvenance

RunStatus = Literal["completed", "failed", "cancelled"]


@dataclass(frozen=True)
class RunArtifacts:
    """Files written once when a run ends."""

    record_path: Path
    output_path: Path | None


class RunSession:
    """Capture immutable source provenance and finalize a run idempotently."""

    def __init__(
        self,
        map_path: str | Path,
        solver_name: str,
        map_root: str | Path = "maps",
        output_root: str | Path = "out",
        records_root: str | Path = "records",
        solver_parameters: Mapping[str, object] | None = None,
        source_bytes: bytes | None = None,
    ) -> None:
        """Read the source fingerprint before simulation begins."""
        self.map_path = Path(map_path).resolve()
        self.map_root = Path(map_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.records_root = Path(records_root).resolve()
        self.solver_name = solver_name
        self.solver_parameters = dict(solver_parameters or {})
        self.run_id = uuid4().hex
        self.timestamp = datetime.now(timezone.utc)
        source = (self.map_path.read_bytes()
                  if source_bytes is None else source_bytes)
        self.map_hash = hashlib.sha256(source).hexdigest()
        self.planning_duration_seconds = 0.0
        self.planning_provenance: PlanningProvenance | None = None
        self._artifacts: RunArtifacts | None = None

    def finalize(
        self,
        simulation: Simulation,
        status: RunStatus = "completed",
        error: str | None = None,
    ) -> RunArtifacts:
        """Save a record; only completed runs replace movement output."""
        if self._artifacts is not None:
            return self._artifacts
        if status == "completed" and not all(
            drone.delivered for drone in simulation.drones
        ):
            raise ValueError("cannot finalize an unfinished run as completed")
        output_path = None
        if status == "completed":
            output_path = OutputWriter(self.output_root).write(
                self.map_path, simulation.history, self.map_root
            )
        filename = self.timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        record_path = self.records_root / f"{filename}_{self.run_id}.json"
        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "map": self.map_path.name,
            "map_path": str(self.map_path),
            "map_sha256": self.map_hash,
            "solver": self.solver_name,
            "solver_parameters": self.solver_parameters,
            "status": status,
            "error": error,
            "planning_duration_seconds": self.planning_duration_seconds,
            "planning": (
                self.planning_provenance
                or PlanningProvenance.capture(simulation.solver)
            ).record(),
            "metrics": asdict(simulation.metrics.summary()),
            "history": [list(turn) for turn in simulation.history],
            "output_path": str(output_path) if output_path else None,
        }
        AtomicFileWriter(record_path).write(
            json.dumps(record, indent=2) + "\n"
        )
        self._artifacts = RunArtifacts(record_path, output_path)
        return self._artifacts
