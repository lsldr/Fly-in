"""Movement, cost, and delivery measurements for simulation runs."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class RunMetrics:
    """Aggregate measurements; arrival averages include delivered drones."""

    turns: int
    total_moves: int
    average_moves_per_turn: float
    total_path_cost: int
    arrival_turns: dict[str, int] = field(default_factory=dict)
    average_turns_per_drone: float = 0.0


class MetricsCollector:
    """Collect movement events, physical edge costs, and delivery turns."""

    def __init__(self) -> None:
        """Initialize an empty run."""
        self._move_counts: list[int] = []
        self._path_cost = 0
        self._arrival_turns: dict[str, int] = {}

    def record_turn(self, moves: list[str]) -> None:
        """Record events, including restricted departures and arrivals."""
        self._move_counts.append(len(moves))

    def record_arrival(self, drone_id: str, turn: int) -> None:
        """Record a drone's first delivery turn, including source waiting."""
        self._arrival_turns.setdefault(drone_id, turn)

    def summary(self) -> RunMetrics:
        """Return a defensive snapshot of collected measurements."""
        turns = len(self._move_counts)
        total_moves = sum(self._move_counts)
        arrivals = dict(self._arrival_turns)
        return RunMetrics(
            turns=turns,
            total_moves=total_moves,
            average_moves_per_turn=total_moves / turns if turns else 0.0,
            total_path_cost=self._path_cost,
            arrival_turns=arrivals,
            average_turns_per_drone=(
                sum(arrivals.values()) / len(arrivals) if arrivals else 0.0
            ),
        )

    def record_path_cost(self, cost: int) -> None:
        """Charge each traversal once using its physical duration."""
        self._path_cost += cost

    def write_record(
        self,
        map_name: str,
        history: list[list[str]],
        directory: str | Path = "records",
        solver_name: str = "greedy",
    ) -> Path:
        """Write a legacy standalone metrics record.

        Application frontends use RunSession for full provenance and status.
        """
        from fly_in.output import AtomicFileWriter

        timestamp = datetime.now(timezone.utc)
        filename = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        record_path = Path(directory) / f"{filename}.json"
        record = {
            "timestamp": timestamp.isoformat(),
            "map": map_name,
            "solver": solver_name,
            "metrics": asdict(self.summary()),
            "history": history,
        }
        AtomicFileWriter(record_path).write(
            json.dumps(record, indent=2) + "\n"
        )
        return record_path
