"""Cross-solver proof records and comparison command outcomes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from fly_in.benchmark import BenchmarkRunner
from fly_in.solver_factory import SOLVER_CHOICES


ROOT = Path(__file__).resolve().parents[1]
DIRECT = 'nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\nconnection: s-g\n'


@pytest.mark.parametrize('strategy', SOLVER_CHOICES)
def test_benchmark_proof_matches_executed_makespan(
    tmp_path: Path, strategy: str,
) -> None:
    """Compare reported evidence with known optima and executed schedules.

    Args:
        tmp_path (Path): Isolated catalog and report destination.
        strategy (str): Registered solver under test.
    """
    maps = tmp_path / 'maps'
    maps.mkdir()
    (maps / 'direct.txt').write_text(DIRECT)
    (maps / 'gap.txt').write_bytes(
        (ROOT / 'tests/fixtures/flow_relaxation_gap.txt').read_bytes(),
    )
    report, saved = BenchmarkRunner(
        maps, strategy, tmp_path / 'records',
        {'time_budget': 5, 'max_iter': 0},
    ).run()
    assert json.loads(saved.read_text()) == report
    rows = report['results']
    assert isinstance(rows, list)
    for row in rows:
        assert row['status'] == 'completed'
        planning = row['planning']
        assert row['search_status'] == planning['status']
        assert row['optimal_makespan'] == planning['optimal_makespan']
        assert planning['optimal_makespan'] == (
            strategy in ('optimal', 'icbs', 'adaptive')
        )
        if strategy == 'greedy':
            assert planning['achieved_makespan'] is None
        else:
            assert planning['achieved_makespan'] == row['turns']
        if planning['optimal_makespan']:
            expected = 1 if row['map'] == 'direct.txt' else 5
            assert row['turns'] == expected
            assert planning['status'] == 'optimal'
            assert planning['proof_reason']
            assert planning['lower_bound'] == (
                1 if row['map'] == 'direct.txt' else 4
            )
            if row['map'] == 'gap.txt':
                assert 'CT' in planning['proof_reason']


@pytest.mark.parametrize('case', ['completed', 'failed', 'target_missed'])
def test_ladder_exit_code_matches_saved_outcomes(
    tmp_path: Path, case: str,
) -> None:
    """The comparison command must not silently succeed on failed runs.

    Args:
        tmp_path (Path): Isolated map catalog and reports.
        case (str): Successful execution, parse failure, or target miss.
    """
    maps = tmp_path / 'maps' / 'easy'
    maps.mkdir(parents=True)
    source = DIRECT
    if case == 'failed':
        source = 'invalid'
    elif case == 'target_missed':
        source = '\n'.join([
            'nb_drones: 1', 'start_hub: h0 0 0', 'end_hub: h7 7 0',
            *(f'hub: h{i} {i} 0' for i in range(1, 7)),
            *(f'connection: h{i}-h{i + 1}' for i in range(7)),
        ])
    (maps / '01_linear_path.txt').write_text(source)
    records = tmp_path / 'records'
    process = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/benchmark_ladder.py'),
         '--map-root', str(maps), '--records-root', str(records),
         '--solver', 'priority', '--export-markdown'],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert process.returncode == (0 if case == 'completed' else 1)
    report_path, = records.glob('ladder_*.json')
    report = json.loads(report_path.read_text())
    assert report['passed'] == (case == 'completed')
    row = report['results'][0]
    assert row['winner'] == ('none' if case == 'failed' else 'priority')
    if case == 'target_missed':
        assert row['strategies']['priority']['meets_target'] is False
        assert 'target missed' in process.stdout
    if case == 'completed':
        assert '(unproved)' in process.stdout
    assert 'Best observed' in process.stdout
