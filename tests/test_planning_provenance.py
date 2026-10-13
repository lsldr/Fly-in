"""Planning certificates remain separate from delivery and failure state."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from fly_in.benchmark import BenchmarkRunner
from fly_in.__main__ import main
from fly_in.gui.main_window import MainWindow
from fly_in.gui.planning import PlanningWorker
from fly_in.parser import Parser
from fly_in.provenance import PlanningProvenance
from fly_in.runs import RunSession
from fly_in.simulation import Simulation
from fly_in.solver_factory import DEFAULT_SOLVER, SolverFactory
from fly_in.solvers.base import ScheduledSolver
from fly_in.solvers.optimal import OptimalSolver, UnprovedPlanningError


TRAP = Path(__file__).parent / 'fixtures' / 'prioritized_trap.txt'
DIRECT = b'nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\nconnection: s-g\n'


def test_factory_default_has_unlimited_proof_limits() -> None:
    """All application defaults construct proof work without hidden cutoffs."""
    assert DEFAULT_SOLVER == 'optimal'
    solver = SolverFactory().create()
    assert isinstance(solver, OptimalSolver)
    assert solver.time_budget is None
    assert solver.node_budget is None
    limited = SolverFactory().create('optimal', {'node_budget': 0})
    assert isinstance(limited, OptimalSolver)
    assert limited.time_budget is None
    assert limited.node_budget == 0


@pytest.mark.parametrize('status', ['failed', 'cancelled'])
def test_execution_failure_preserves_successful_planning_proof(
    tmp_path: Path, status: str,
) -> None:
    """Preserve a completed proof when execution later stops.

    Args:
        tmp_path (Path): Isolated directory for run records.
        status (str): Execution outcome, either failed or cancelled.
    """
    solver = SolverFactory().create()
    assert isinstance(solver, ScheduledSolver)
    simulation = Simulation.from_config(
        Parser().parse_text(DIRECT.decode()), solver,
    )
    solver.prepare(simulation.network, simulation.drones)
    session = RunSession(
        tmp_path / 'direct.txt', DEFAULT_SOLVER,
        records_root=tmp_path / 'records', source_bytes=DIRECT,
    )
    artifacts = session.finalize(
        simulation, 'failed' if status == 'failed' else 'cancelled',
    )
    record = json.loads(artifacts.record_path.read_text())
    assert record['status'] == status
    assert record['history'] == []
    assert record['output_path'] is None
    assert record['planning'] == {
        'optimal_makespan': True,
        'status': 'optimal',
        'proof_reason': solver.solution.reason if solver.solution else None,
        'lower_bound': 1,
        'achieved_makespan': 1,
    }


def test_completed_heuristic_run_does_not_invent_a_proof(
    tmp_path: Path,
) -> None:
    """Keep heuristic completion separate from an optimality proof.

    Args:
        tmp_path (Path): Isolated directory for output and run records.
    """
    simulation = Simulation.from_config(
        Parser().parse_text(DIRECT.decode()),
        SolverFactory().create('priority'),
    )
    simulation.run()
    session = RunSession(
        tmp_path / 'direct.txt', 'priority', records_root=tmp_path / 'records',
        output_root=tmp_path / 'out', source_bytes=DIRECT,
    )
    artifacts = session.finalize(simulation)
    record = json.loads(artifacts.record_path.read_text())
    assert record['status'] == 'completed'
    assert record['solver'] == 'priority'
    assert record['planning']['optimal_makespan'] is False
    assert record['planning']['status'] == 'feasible'
    assert record['planning']['achieved_makespan'] == 1


def test_unproved_incumbent_is_diagnostic_only() -> None:
    """A bounded legal incumbent has measurements but never an optimal flag."""
    simulation = Simulation.from_config(
        Parser().parse(TRAP), OptimalSolver(node_budget=0),
    )
    solver = simulation.solver
    assert isinstance(solver, OptimalSolver)
    with pytest.raises(UnprovedPlanningError) as failure:
        solver.prepare(simulation.network, simulation.drones)
    planning = PlanningProvenance.capture(solver, failure.value).record()
    assert solver.solution is None
    assert planning['optimal_makespan'] is False
    assert planning['status'] == 'node_limit'
    assert planning['lower_bound'] == 3
    assert planning['achieved_makespan'] == 4
    assert planning['proof_reason']


def test_cli_unproved_timeout_has_failure_record_and_no_moves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Record proof timeout before playback and leave movement output absent.

    Args:
        tmp_path (Path): Isolated root for output and record directories.
        capsys (pytest.CaptureFixture[str]): Captured CLI stdout and stderr.
    """
    records = tmp_path / 'records'
    output = tmp_path / 'out'
    with pytest.raises(SystemExit) as failure:
        main([str(TRAP), '--solver', 'optimal', '--time-budget', '0',
              '--display', 'moves', '--records-dir', str(records),
              '--output-dir', str(output)])
    assert failure.value.code != 0
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'unproved' in captured.err
    assert not output.exists()
    paths = list(records.glob('*.json'))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record['status'] == 'failed'
    assert record['history'] == []
    assert record['planning']['status'] == 'timeout'
    assert record['planning']['optimal_makespan'] is False


def test_benchmark_unproved_floor_is_not_completed(tmp_path: Path) -> None:
    """Keep an interrupted benchmark distinct from completed execution.

    Args:
        tmp_path (Path): Isolated map catalog and report root.
    """
    maps = tmp_path / 'maps'
    maps.mkdir()
    (maps / TRAP.name).write_bytes(TRAP.read_bytes())
    report, _ = BenchmarkRunner(
        maps, 'optimal', tmp_path / 'records', {'node_budget': 0},
    ).run()
    rows = report['results']
    assert isinstance(rows, list)
    row = rows[0]
    assert row['status'] == 'failed'
    assert 'turns' not in row
    assert row['planning']['status'] == 'node_limit'
    assert row['planning']['achieved_makespan'] == 4
    assert row['planning']['optimal_makespan'] is False
    assert row['optimal_makespan'] is False
    assert row['search_status'] == 'node_limit'


def test_gui_unproved_record_and_disabled_playback(
    tmp_path: Path, app: QApplication,
) -> None:
    """Preserve diagnostics and disable playback after planning failure.

    Args:
        tmp_path (Path): Isolated directory for records and output.
        app (QApplication): Shared Qt application fixture.
    """
    with patch.object(PlanningWorker, 'start', return_value=None):
        window = MainWindow(
            TRAP.parent, initial_map=TRAP, solver_name='optimal',
            solver_parameters={'node_budget': 0},
            records_dir=tmp_path / 'records', output_dir=tmp_path / 'out',
            animate=False,
        )
        try:
            worker = window._workers[-1]
            prepared: list[int] = []
            worker.prepared.connect(lambda generation, result, elapsed:
                                    prepared.append(generation))
            worker.run()
            app.processEvents()
            assert prepared == []
            assert not window._ready
            assert not window.play_button.isEnabled()
            assert not window.step_button.isEnabled()
            assert window.reset_button.isEnabled()
            record_path, = (tmp_path / 'records').glob('*.json')
            record = json.loads(record_path.read_text())
            assert record['status'] == 'failed'
            assert record['planning']['status'] == 'node_limit'
            assert record['planning']['lower_bound'] == 3
            assert record['planning']['achieved_makespan'] == 4
            assert record['planning']['optimal_makespan'] is False
            assert record['history'] == []
            assert not (tmp_path / 'out').exists()
        finally:
            window.close()


def test_gui_adoption_rejects_unproved_proof_mode_result(
    tmp_path: Path, app: QApplication,
) -> None:
    """Reject an unproved prepared result in proof-mode playback.

    Args:
        tmp_path (Path): Isolated directory for records and output.
        app (QApplication): Shared Qt application fixture.
    """
    with patch.object(PlanningWorker, 'start', return_value=None):
        window = MainWindow(
            TRAP.parent, initial_map=TRAP, solver_name='optimal',
            records_dir=tmp_path / 'records', output_dir=tmp_path / 'out',
            animate=False,
        )
        try:
            simulation = Simulation.from_config(
                Parser().parse(TRAP), SolverFactory().create('priority'),
            )
            solver = simulation.solver
            assert isinstance(solver, ScheduledSolver)
            solver.prepare(simulation.network, simulation.drones)
            window._prepared(window._generation, simulation, 0.1)
            assert not window._ready
            assert not window.play_button.isEnabled()
            assert not window.step_button.isEnabled()
            assert window.reset_button.isEnabled()
        finally:
            window.close()


def test_execution_error_keeps_cli_planning_certificate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the planning certificate when terminal playback fails.

    Args:
        tmp_path (Path): Isolated root for source and records.
        capsys (pytest.CaptureFixture[str]): Captured CLI output streams.
    """
    source = tmp_path / 'direct.txt'
    source.write_bytes(DIRECT)
    records = tmp_path / 'records'
    with patch.object(Simulation, 'run', side_effect=RuntimeError('playback')):
        with pytest.raises(SystemExit):
            main([str(source), '--display', 'moves',
                  '--records-dir', str(records),
                  '--output-dir', str(tmp_path / 'out')])
    record_path, = records.glob('*.json')
    record = json.loads(record_path.read_text())
    assert record['status'] == 'failed'
    assert record['error'] == 'playback'
    assert record['planning']['optimal_makespan'] is True
    assert record['planning']['status'] == 'optimal'
    assert record['planning']['lower_bound'] == 1
    assert record['planning']['achieved_makespan'] == 1
    assert capsys.readouterr().out == ''


def test_execution_error_keeps_benchmark_planning_certificate(
    tmp_path: Path,
) -> None:
    """Keep the planning certificate when benchmark execution fails.

    Args:
        tmp_path (Path): Isolated map catalog and report root.
    """
    maps = tmp_path / 'maps'
    maps.mkdir()
    (maps / 'direct.txt').write_bytes(DIRECT)
    with patch.object(Simulation, 'run', side_effect=RuntimeError('playback')):
        report, _ = BenchmarkRunner(
            maps, records_root=tmp_path / 'records',
        ).run()
    rows = report['results']
    assert isinstance(rows, list)
    assert rows[0]['status'] == 'failed'
    assert rows[0]['error'] == 'playback'
    assert rows[0]['planning']['optimal_makespan'] is True
    assert rows[0]['planning']['status'] == 'optimal'
    assert rows[0]['optimal_makespan'] is True
    assert rows[0]['search_status'] == 'optimal'
