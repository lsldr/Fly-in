"""Benchmark discovery, provenance, targets, and failure handling."""

import json
from pathlib import Path
from unittest.mock import patch

from fly_in.benchmark import BenchmarkRunner
from fly_in.metrics import RunMetrics
from fly_in.simulation import SimulationResult


SOURCE = 'nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\nconnection: s-g'


def test_benchmark_discovers_only_text_and_records_failures(
    tmp_path: Path,
) -> None:
    """Continue the catalog after a malformed map and save a report.

    Args:
        tmp_path (Path): Isolated catalog and report root.
    """
    root = tmp_path / 'maps'
    root.mkdir()
    (root / 'good.txt').write_text(SOURCE)
    (root / 'bad.txt').write_text('invalid')
    (root / 'ignored.map').write_text('invalid')
    report, path = BenchmarkRunner(
        root, records_root=tmp_path / 'records',
    ).run()
    rows = report['results']
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert rows[0]['status'] == 'failed'
    assert rows[1]['status'] == 'completed'
    assert rows[1]['turns'] == 1
    assert rows[1]['map_sha256']
    assert json.loads(path.read_text()) == report


def test_challenger_requires_strictly_better_than_45(tmp_path: Path) -> None:
    """Require a challenger result strictly below its reference target.

    Args:
        tmp_path (Path): Isolated catalog and report root.
    """
    root = tmp_path / 'maps'
    (root / 'challenger').mkdir(parents=True)
    (root / 'challenger' / '01_the_impossible_dream.txt').write_text(SOURCE)
    result = SimulationResult(45, [], RunMetrics(45, 1, 1 / 45, 1))
    with patch('fly_in.benchmark.Simulation.run', return_value=result):
        report, _ = BenchmarkRunner(
            root, records_root=tmp_path / 'records',
        ).run()
    rows = report['results']
    assert isinstance(rows, list)
    assert rows[0]['target_turns'] == 45
    assert rows[0]['meets_target'] is False
