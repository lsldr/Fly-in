"""Default routing must minimize turns, beyond meeting benchmark targets."""

from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from joint_oracle import exact_makespan
from fly_in.benchmark import TARGETS
from fly_in.__main__ import main
from fly_in.cli import CliApplication
from fly_in.solver_factory import SolverFactory
from fly_in.gui.main_window import MainWindow
from fly_in.gui.planning import PlanningWorker
from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solvers.base import ScheduledSolver
from fly_in.solvers.bounds import NetworkProfiler
from fly_in.solvers.icbs import ICBSSolver


ROOT = Path(__file__).resolve().parents[1]
TRAP = ROOT / "tests/fixtures/prioritized_trap.txt"
CASES = [
    pytest.param(TRAP.read_text(encoding="utf-8"), 3, id="priority-trap"),
    pytest.param("""nb_drones: 3
start_hub: s 0 0 [max_drones=0]
end_hub: g 1 0 [max_drones=invalid]
connection: s-g [max_link_capacity=2]
""", 2, id="unlimited-endpoints"),
    pytest.param("""nb_drones: 3
start_hub: s 0 0
end_hub: g 2 0
hub: a 1 0 [max_drones=2]
connection: s-a [max_link_capacity=3]
connection: a-g [max_link_capacity=3]
""", 3, id="hub-capacity-two"),
    pytest.param("""nb_drones: 3
start_hub: s 0 0
end_hub: g 2 0
hub: r 1 0 [zone=restricted]
connection: s-r
connection: r-g
""", 5, id="restricted-arrival-release"),
    pytest.param("""nb_drones: 3
start_hub: s 0 0
end_hub: g 1 0 [zone=restricted]
connection: s-g
""", 4, id="restricted-goal"),
    pytest.param("""nb_drones: 2
start_hub: s 0 0
end_hub: g 2 0
hub: x 1 0 [zone=blocked]
hub: a 1 1
hub: dead 0 1
hub: isolated 9 9
connection: s-x [max_link_capacity=9]
connection: x-g [max_link_capacity=9]
connection: s-a
connection: a-g
connection: s-dead
""", 3, id="blocked-dead-end-disconnected-hub"),
    pytest.param("""nb_drones: 3
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0
hub: b 1 1 [zone=restricted]
hub: c 2 1
connection: s-a
connection: a-g
connection: s-b
connection: b-c
connection: c-g
""", 4, id="unequal-route-latencies"),
    pytest.param("""nb_drones: 3
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0
hub: b 2 0
connection: s-a
connection: a-b
connection: b-s
connection: b-g
""", 4, id="cycle-and-shared-link"),
]


def _default_solver_name() -> str:
    """Resolve the configured CLI solver default.

    Returns:
        str: Solver key parsed from empty CLI arguments.
    """
    return str(CliApplication().argument_parser().parse_args([]).solver)


@pytest.mark.parametrize("text,expected", CASES)
def test_default_matches_independent_joint_optimum(
    text: str, expected: int, tmp_path: Path,
) -> None:
    """Compare default-solver delivery with an independent state search.

    Args:
        text (str): Map text for one topology and capacity case.
        expected (int): Independently established optimal makespan.
        tmp_path (Path): Temporary records directory for solver construction.
    """
    config = Parser().parse_text(text)
    optimum = exact_makespan(DroneNetwork.from_config(config),
                             config.nb_drones)
    assert optimum == expected
    solver = SolverFactory(records_dir=tmp_path).create(_default_solver_name())
    simulation = Simulation.from_config(config, solver)
    result = simulation.run(max_turns=30)
    assert all(drone.delivered for drone in simulation.drones)
    assert result.turns == optimum


def test_default_provides_a_makespan_certificate(tmp_path: Path) -> None:
    """Require the default solver to attach an explicit proof.

    Args:
        tmp_path (Path): Temporary records directory for solver construction.
    """
    solver = SolverFactory(records_dir=tmp_path).create(_default_solver_name())
    assert isinstance(solver, ScheduledSolver)
    simulation = Simulation.from_config(Parser().parse(TRAP), solver)
    solver.prepare(simulation.network, simulation.drones)
    assert solver.solution is not None
    assert solver.solution.optimal, solver.solution.reason
    assert solver.solution.status == "optimal"
    assert solver.solution.reason


def test_icbs_certifies_trap_and_budgeted_cutoffs_stay_honest() -> None:
    """The counterexample really needs search; cutoffs cannot imply proof."""
    config = Parser().parse(TRAP)
    network = DroneNetwork.from_config(config)
    assert exact_makespan(network, config.nb_drones) == 3
    proved = ICBSSolver(time_budget=10).solve(network, ["D1", "D2"])
    assert proved.makespan == 3
    assert proved.optimal
    assert proved.status == "optimal"
    for solver, status in (
        (ICBSSolver(time_budget=0), "timeout"),
        (ICBSSolver(time_budget=10, node_budget=0), "node_limit"),
    ):
        simulation = Simulation.from_config(config, solver)
        solver.prepare(simulation.network, simulation.drones)
        assert solver.solution is not None
        assert not solver.solution.optimal
        assert solver.solution.status == status
        assert simulation.run(max_turns=10).turns == 4


def test_cli_without_solver_flag_delivers_trap_in_three_turns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise CLI defaults, terminal output, and persistence together.

    Args:
        tmp_path (Path): Isolated output and records root.
        capsys (pytest.CaptureFixture[str]): Captured terminal streams.
    """
    main([str(TRAP), "--display", "moves", "--map-root", str(TRAP.parent),
          "--output-dir", str(tmp_path / "out"),
          "--records-dir", str(tmp_path / "records")])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert sorted(lines[-1].split()) == ["D1-g", "D2-g"]
    assert (tmp_path / "out" / TRAP.name).read_text().splitlines() == lines


def test_gui_without_solver_override_delivers_trap_in_three_turns(
    tmp_path: Path, app: QApplication,
) -> None:
    """Exercise desktop default selection and three-turn delivery.

    Args:
        tmp_path (Path): Isolated output and records root.
        app (QApplication): Shared Qt application fixture.
    """
    with patch.object(PlanningWorker, "start", return_value=None):
        window = MainWindow(
            TRAP.parent, initial_map=TRAP, animate=False,
            records_dir=tmp_path / "records", output_dir=tmp_path / "out",
        )
        try:
            window._workers[-1].run()
            app.processEvents()
            assert window._ready
            for _ in range(3):
                window._step_simulation()
            assert window.simulation is not None
            assert all(drone.delivered for drone in window.simulation.drones)
            assert window.progress.value() == 100
        finally:
            window.close()


@pytest.mark.parametrize("path", [ROOT / "maps" / name
                                  for name in sorted(TARGETS)],
                         ids=lambda path: path.relative_to(ROOT).as_posix())
def test_default_supplied_maps_match_certified_flow_bounds(
    path: Path, tmp_path: Path,
) -> None:
    """Match each supplied map's certified flow lower bound.

    Args:
        path (Path): Catalog map under test.
        tmp_path (Path): Temporary records directory for solver construction.
    """
    config = Parser().parse(path)
    # Size is a test envelope, not a claim that size bounds search difficulty.
    assert config.nb_drones <= 25
    assert len(config.zones) + 2 <= 54
    assert len(config.connections) <= 70
    solver = SolverFactory(records_dir=tmp_path).create(_default_solver_name())
    simulation = Simulation.from_config(config, solver)
    bound = NetworkProfiler(simulation.network).profile(
        config.nb_drones,
    ).lower_bound
    result = simulation.run(max_turns=100)
    assert all(drone.delivered for drone in simulation.drones)
    assert result.turns == bound
