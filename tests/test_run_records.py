"""Shared persistence and immutable source provenance regressions."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fly_in.output import AtomicFileWriter, OutputWriter
from fly_in.parser import Parser
from fly_in.runs import RunSession
from fly_in.simulation import Simulation
from fly_in.solvers.prioritized import PriorityBasedSolver

SOURCE = b'''nb_drones: 1
start_hub: s 0 0
end_hub: g 2 0
hub: r 1 0 [zone=restricted]
connection: s-r
connection: r-g
'''


def simulation() -> Simulation:
    """Build a fresh simulation for a restricted-link traversal.

    Returns:
        Simulation: Two-drone run state before its first turn.
    """
    return Simulation.from_config(Parser().parse_text(SOURCE.decode()),
                                  PriorityBasedSolver())


def test_completed_record_is_idempotent_and_tracks_arrivals(
    tmp_path: Path,
) -> None:
    """Finalize once and retain movement and arrival measurements.

    Args:
        tmp_path (Path): Isolated root for source and generated artifacts.
    """
    run = RunSession(tmp_path / 'maps' / 'easy' / 'a.txt', 'priority',
                     tmp_path / 'maps', tmp_path / 'out', tmp_path / 'records',
                     source_bytes=SOURCE)
    sim = simulation()
    sim.run()
    run.planning_duration_seconds = 0.125
    artifacts = run.finalize(sim)
    assert run.finalize(sim) is artifacts
    record = json.loads(artifacts.record_path.read_text())
    assert record['map_sha256'] == hashlib.sha256(SOURCE).hexdigest()
    assert record['planning_duration_seconds'] == 0.125
    assert record['metrics']['total_path_cost'] == 3
    assert record['metrics']['arrival_turns'] == {'D1': 3}
    assert record['metrics']['average_turns_per_drone'] == 3
    assert artifacts.output_path == tmp_path / 'out' / 'easy' / 'a.txt'
    assert artifacts.output_path.read_text() == 'D1-r-s\nD1-r\nD1-g\n'
    assert len(list((tmp_path / 'records').glob('*.json'))) == 1


@pytest.mark.parametrize('status', ['failed', 'cancelled'])
def test_partial_record_preserves_successful_output(
    tmp_path: Path, status: str,
) -> None:
    """Keep prior movement output when recording a partial run.

    Args:
        tmp_path (Path): Isolated filesystem root for both runs.
        status (str): Partial run state, such as failed or cancelled.
    """
    source = tmp_path / 'maps' / 'a.txt'
    output = OutputWriter(tmp_path / 'out').write(
        source, [['previous']], tmp_path / 'maps')
    run = RunSession(source, 'priority', tmp_path / 'maps', tmp_path / 'out',
                     tmp_path / 'records', source_bytes=SOURCE)
    sim = simulation()
    sim.step()
    with pytest.raises(ValueError, match='unfinished'):
        run.finalize(sim)
    assert status in ('failed', 'cancelled')
    artifacts = run.finalize(
        sim, 'failed' if status == 'failed' else 'cancelled')
    assert artifacts.output_path is None
    assert json.loads(artifacts.record_path.read_text())['status'] == status
    assert output.read_text() == 'previous\n'


def test_external_names_do_not_collide(tmp_path: Path) -> None:
    """Distinguish external maps that share a basename.

    Args:
        tmp_path (Path): Temporary root containing separate source trees.
    """
    writer = OutputWriter(tmp_path / 'out')
    first = writer.write(tmp_path / 'first' / 'a.txt', [['a']])
    second = writer.write(tmp_path / 'second' / 'a.txt', [['b']])
    assert first != second
    assert first.read_text() == 'a\n'
    assert second.read_text() == 'b\n'


def test_failed_replace_keeps_old_file_and_cleans_temporary(
    tmp_path: Path,
) -> None:
    """Preserve old contents and remove the temporary file after failure.

    Args:
        tmp_path (Path): Temporary directory for the target and temp file.
    """
    output = tmp_path / 'output.txt'
    output.write_text('old')
    with patch.object(Path, 'replace', side_effect=OSError('denied')):
        with pytest.raises(OSError, match='denied'):
            AtomicFileWriter(output).write('new')
    assert output.read_text() == 'old'
    assert list(tmp_path.iterdir()) == [output]
