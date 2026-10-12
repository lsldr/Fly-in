"""Movement, cost, and delivery measurements for simulation runs."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class RunMetrics:
    """Aggregate measurements for one run.

    Arrival averages include only drones with recorded delivery turns.

    Attributes:
        turns (int): Number of turns recorded.
        total_moves (int): Number of movement events, including transit
            departures and arrivals.
        average_moves_per_turn (float): Mean recorded events per turn.
        total_path_cost (int): Sum of physical traversal durations.
        arrival_turns (dict[str, int]): First delivery turn by drone ID.
        average_turns_per_drone (float): Mean delivery turn among arrivals.
    """

    turns: int
    total_moves: int
    average_moves_per_turn: float
    total_path_cost: int
    arrival_turns: dict[str, int] = field(default_factory=dict)
    average_turns_per_drone: float = 0.0


class MetricsCollector:
    """Collector for movement events, physical costs, and delivery turns."""

    def __init__(self) -> None:
        """Initialize empty movement, cost, and arrival measurements."""
        self._move_counts: list[int] = []
        self._path_cost = 0
        self._arrival_turns: dict[str, int] = {}

    def record_turn(self, moves: list[str]) -> None:
        """Record the number of events emitted during a turn.

        Args:
            moves (list[str]): Movement events for the turn. Restricted-link
                departures and their later arrivals each count as events.
        """
        self._move_counts.append(len(moves))

    def record_arrival(self, drone_id: str, turn: int) -> None:
        """Keep the first recorded delivery turn for a drone.

        Args:
            drone_id (str): Identifier of the delivered drone.
            turn (int): One-based simulation turn of delivery, including any
                time spent waiting at the source.
        """
        self._arrival_turns.setdefault(drone_id, turn)

    def summary(self) -> RunMetrics:
        """Return a snapshot of collected measurements.

        Returns:
            RunMetrics: Aggregate values with a copied arrival mapping.
        """
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
        """Add one traversal's physical duration to the total cost.

        Args:
            cost (int): Movement duration charged for a traversal.
        """
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

        Args:
            map_name (str): Map label stored in the record.
            history (list[list[str]]): Per-turn movement event history.
            directory (str | Path): Destination directory for the JSON file.
            solver_name (str): Strategy label stored with the measurements.

        Returns:
            Path: Path to the atomically written JSON record.
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
