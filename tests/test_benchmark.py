"""Benchmark discovery, provenance, targets, and failure handling."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fly_in.benchmark import BenchmarkApplication, BenchmarkRunner
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


@pytest.mark.parametrize("turns,meets_target", [
    (43, True), (44, False), (45, False),
])
@pytest.mark.parametrize("category_root", [False, True])
def test_challenger_requires_version_two_optimum(
    tmp_path: Path, turns: int, meets_target: bool, category_root: bool,
) -> None:
    """Version two requires 43 turns from either catalog entry root.

    Args:
        tmp_path (Path): Isolated catalog and report root.
        turns (int): Achieved turn count to assess against the v2 target.
        meets_target (bool): Whether the result meets the required optimum.
        category_root (bool): Whether discovery starts inside challenger.
    """
    root = tmp_path / 'maps'
    (root / 'challenger').mkdir(parents=True)
    (root / 'challenger' / '01_the_impossible_dream.txt').write_text(SOURCE)
    result = SimulationResult(turns, [], RunMetrics(turns, 1, 1 / turns, 1))
    with patch('fly_in.benchmark.Simulation.run', return_value=result):
        report, _ = BenchmarkRunner(
            root / 'challenger' if category_root else root,
            records_root=tmp_path / 'records',
        ).run()
    rows = report['results']
    assert isinstance(rows, list)
    assert rows[0]['target_turns'] == 43
    assert rows[0]['meets_target'] is meets_target


def test_cli_announces_map_before_planning_and_reports_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose the active map before a slow or failing preparation begins.

    Args:
        tmp_path (Path): Isolated catalog and report destination.
        capsys (pytest.CaptureFixture[str]): Captured CLI output.
    """
    root = tmp_path / 'maps'
    root.mkdir()
    (root / 'slow.txt').write_text(SOURCE)

    def fail_planning(*args: object) -> None:
        """Check immediate visibility, then simulate a planning failure.

        Args:
            *args (object): Unused preparation inputs.
        """
        assert '[1/1] Starting slow.txt' in capsys.readouterr().out
        raise RuntimeError('planning interrupted')

    with patch('fly_in.solvers.optimal.OptimalSolver.prepare',
               side_effect=fail_planning):
        status = BenchmarkApplication().run([
            '--map-root', str(root), '--records-root',
            str(tmp_path / 'records'), '--solver', 'optimal',
        ])
    output = capsys.readouterr().out
    assert status == 1
    assert '[1/1] slow.txt: failed' in output
    assert 'planning interrupted' in output
    assert 'Report:' in output
