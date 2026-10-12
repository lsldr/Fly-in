"""Independent exact, interruption, and analytic-envelope proof contracts."""

from collections.abc import Iterable
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from random import Random
from threading import Event
from time import perf_counter
import tracemalloc
from typing import Callable

import pytest
from PySide6.QtWidgets import QApplication

from joint_oracle import exact_makespan
from fly_in.benchmark import TARGETS
from fly_in.gui.planning import PlanningWorker
from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solvers.base import (
    ConflictDetector, Constraint, PlanningCancelled, ReservationTable,
    SearchBudget, SearchTimeout, Solution, SpaceTimeAStar,
    TimedMove, TimedPath,
)
from fly_in.solvers.bounds import NetworkProfiler
from fly_in.solvers.optimal import OptimalSolver, UnprovedPlanningError
from fly_in.solvers.prioritized import PrioritizedPlanner


ROOT = Path(__file__).resolve().parents[1]
TRAP = ROOT / "tests/fixtures/prioritized_trap.txt"
FLOW_GAP = ROOT / "tests/fixtures/flow_relaxation_gap.txt"


def _network(text: str) -> DroneNetwork:
    """Parse each independent declaration into a fresh topology.

    Args:
        text (str): Map declarations to parse into a fresh test topology.

    Returns:
        DroneNetwork: Fresh topology parsed from the supplied map text.
    """
    return DroneNetwork.from_config(Parser().parse_text(text))


def _ids(fleet: int) -> list[str]:
    """Provide conventional IDs without using solver ordering helpers.

    Args:
        fleet (int): Number of drone identifiers to generate.

    Returns:
        list[str]: Conventional identifiers from D1 through the fleet size.
    """
    return [f"D{index + 1}" for index in range(fleet)]


def _measured_solve(
    solver: OptimalSolver, network: DroneNetwork, fleet: int,
    initial: Solution | None,
    record_property: Callable[[str, object], None],
) -> Solution:
    """Attach search time and Python peak allocation to the pytest report.

    Args:
        solver (OptimalSolver): Exact solver whose planning time and peak
            Python allocation are measured.
        network (DroneNetwork): Topology passed to the exact solver.
        fleet (int): Number of drones included in the solve request.
        initial (Solution | None): Feasible incumbent supplied to the search or
            measured solve.
        record_property (Callable[[str, object], None]): Pytest report fixture
            used to attach planning measurements.

    Returns:
        Solution: Proved result from the timed exact search.
    """
    tracemalloc.start()
    try:
        started = perf_counter()
        result = solver.solve(network, _ids(fleet), initial)
        seconds = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    record_property("planning_seconds", seconds)
    record_property("planning_peak_bytes", peak)
    return result


def _serial_seed(network: DroneNetwork, fleet: int) -> Solution:
    """Separate independent paths completely to obtain a loose legal floor.

    Args:
        network (DroneNetwork): Topology used to construct independent paths.
        fleet (int): Number of drone paths separated with start-hub waits.

    Returns:
        Solution: Feasible paths separated by waits to provide a loose floor.
    """
    engine = SpaceTimeAStar(network)
    paths = {}
    offset = 0
    for agent in _ids(fleet):
        path = engine.find_path(agent)
        assert path is not None
        states = tuple((network.start.name, turn) for turn in range(offset))
        states += tuple((name, turn + offset) for name, turn in path.states)
        moves = tuple(TimedMove(
            move.source, move.destination, move.link,
            move.departure + offset, move.arrival + offset,
        ) for move in path.moves)
        paths[agent] = TimedPath(agent, states, moves)
        offset += path.arrival
    return Solution(paths)


def _mixed_text(seed: int) -> str:
    """Mix cycles, capacities, priorities, and two-turn destination costs.

    Args:
        seed (int): Fixed random seed used to generate this map.

    Returns:
        str: Map declaration text generated from the fixed random seed.
    """
    random = Random(seed)
    names = ["s", "a", "b", "c", "g"]
    rows = ["nb_drones: 3", "start_hub: s 0 0", "end_hub: g 4 0"]
    for index, name in enumerate(names[1:-1], 1):
        kind = random.choice(["normal", "priority", "restricted"])
        capacity = random.choice([1, 2])
        rows.append(f"hub: {name} {index} 0 "
                    f"[zone={kind} max_drones={capacity}]")
    edges = {("s", "a"), ("a", "b"), ("b", "g")}
    for first, second in combinations(names, 2):
        if random.random() < 0.3:
            edges.add((first, second))
    for first, second in sorted(edges):
        capacity = random.choice([1, 2])
        rows.append(f"connection: {first}-{second} "
                    f"[max_link_capacity={capacity}]")
    return "\n".join(rows) + "\n"


def _permuted(text: str) -> str:
    """Reverse hub/link declarations while preserving endpoint declarations.

    Args:
        text (str): Map declaration text used to build a fresh test topology.

    Returns:
        str: Map text with hub and connection declaration order reversed.
    """
    rows = text.splitlines()
    headers = [row for row in rows if not row.startswith(
        ("hub:", "connection:")
    )]
    hubs = [row for row in rows if row.startswith("hub:")]
    links = [row for row in rows if row.startswith("connection:")]
    return "\n".join(headers + list(reversed(hubs))
                     + list(reversed(links))) + "\n"


@pytest.mark.parametrize("seed", [3, 17, 31, 47, 59, 71, 89, 103])
@pytest.mark.parametrize("permuted", [False, True])
def test_mixed_topologies_match_independent_bfs(
    seed: int, permuted: bool,
) -> None:
    """Certify exact makespans for tiny fixed-seed topologies.

    Args:
        seed (int): Fixed random seed used to generate this map.
        permuted (bool): Whether declarations are reversed before parsing to
            check order invariance.
    """
    text = _mixed_text(seed)
    network = _network(_permuted(text) if permuted else text)
    expected = exact_makespan(network, 3)
    result = OptimalSolver().solve(network, _ids(3))
    assert result.makespan == expected
    assert result.optimal and result.status == "optimal" and result.reason
    assert result.lower_bound is not None
    assert result.lower_bound <= expected
    assert not ConflictDetector(network).find(result.paths)


@pytest.mark.parametrize("path", [TRAP, FLOW_GAP], ids=["trap", "flow-gap"])
@pytest.mark.parametrize("permuted", [False, True])
def test_complete_ct_search_including_nontight_flow_bound(
    path: Path, permuted: bool,
    record_property: Callable[[str, object], None],
) -> None:
    """A real CT certificate must work when aggregate flow is insufficient.

    Args:
        path (Path): Map path selected from fixture or supplied-catalog cases.
        permuted (bool): Whether declarations are reversed before parsing to
            check order invariance.
        record_property (Callable[[str, object], None]): Pytest report fixture
            used to attach planning measurements.
    """
    text = path.read_text(encoding="utf-8")
    config = Parser().parse_text(_permuted(text) if permuted else text)
    network = DroneNetwork.from_config(config)
    expected = exact_makespan(network, config.nb_drones)
    lower = NetworkProfiler(network).profile(config.nb_drones).lower_bound
    if path == FLOW_GAP:
        assert (lower, expected) == (4, 5)
    else:
        assert (lower, expected) == (3, 3)
    initial = _serial_seed(network, config.nb_drones)
    result = _measured_solve(OptimalSolver(), network, config.nb_drones,
                             initial, record_property)
    assert initial.makespan > expected
    assert result.makespan == expected
    assert result.optimal and result.status == "optimal"
    assert result.stats.nodes > 0
    assert result.reason


@pytest.mark.parametrize("path", [ROOT / "maps" / name
                                  for name in sorted(TARGETS)],
                         ids=lambda path: path.relative_to(ROOT).as_posix())
def test_supplied_map_certificates_and_planning_measurements(
    path: Path, record_property: Callable[[str, object], None],
) -> None:
    """Measure current proof planning under the supplied catalog identities.

    Args:
        path (Path): Map path selected from fixture or supplied-catalog cases.
        record_property (Callable[[str, object], None]): Pytest report fixture
            used to attach planning measurements.
    """
    config = Parser().parse(path)
    network = DroneNetwork.from_config(config)
    bound = NetworkProfiler(network).profile(config.nb_drones).lower_bound
    result = _measured_solve(OptimalSolver(), network, config.nb_drones,
                             None, record_property)
    assert result.makespan == bound
    assert result.optimal and result.status == "optimal" and result.reason
    assert result.lower_bound == bound
    assert not ConflictDetector(network).find(result.paths)
    if config.nb_drones == 25:
        assert (len(network.zones), len(network.connections), bound) == (
            54, 70, 43,
        )


class EnvelopeMaps:
    """Build deterministic families with constructive matching lower bounds."""

    @staticmethod
    def chain(
        length: int, capacity: int = 1, restricted: bool = False,
    ) -> list[str]:
        """Declare one pipeline with unlimited endpoints.

        Args:
            length (int): Number of connections in the generated pipeline.
            capacity (int): Capacity shared by generated intermediate hubs and
                connections.
            restricted (bool): Whether intermediate hubs have two-turn entry
                duration.

        Returns:
            list[str]: Pipeline map declarations with the requested capacity.
        """
        rows = ["nb_drones: 25", "start_hub: s 0 0",
                f"end_hub: g {length} 0"]
        names = ["s"] + [f"h{i}" for i in range(1, length)] + ["g"]
        for index, name in enumerate(names[1:-1], 1):
            kind = "restricted" if restricted else "normal"
            rows.append(f"hub: {name} {index} 0 "
                        f"[zone={kind} max_drones={capacity}]")
        for first, second in zip(names, names[1:]):
            rows.append(f"connection: {first}-{second} "
                        f"[max_link_capacity={capacity}]")
        return rows

    @classmethod
    def build(cls, family: str) -> tuple[str, int]:
        """Return an envelope map and its matching analytic makespan.

        Args:
            family (str): Analytic topology family selected for lower-bound
                verification.

        Returns:
            tuple[str, int]: Map text and its analytic makespan.
        """
        if family == "capacity-three":
            rows = cls.chain(9, capacity=3)
            expected = 9 + (25 + 2) // 3 - 1
        elif family == "restricted-chain":
            rows = cls.chain(6, restricted=True)
            expected = 11 + 25 - 1
        elif family == "cyclic-bottleneck":
            rows = cls.chain(10)
            rows += ["hub: x 4 1", "hub: y 4 2", "connection: h4-x",
                     "connection: x-y", "connection: y-h4"]
            expected = 10 + 25 - 1
        elif family == "unusable-full-envelope":
            rows = cls.chain(19)
            for index in range(34):
                kind = "blocked" if index % 3 == 0 else "normal"
                rows.append(f"hub: u{index} {index} 1 [zone={kind}]")
                rows.append(f"connection: s-u{index}")
            for index in range(17):
                rows.append(f"connection: u{2 * index}-u{2 * index + 1}")
            expected = 19 + 25 - 1
        else:
            shared = family == "narrow-final-cut"
            branches = 4 if shared else 5
            rows = ["nb_drones: 25", "start_hub: s 0 0", "end_hub: g 6 0"]
            if shared:
                rows += ["hub: choke 5 0", "connection: choke-g"]
            for branch in range(branches):
                names = ["s"]
                for step in range(1, 5 if shared else 6):
                    name = f"p{branch}x{step}"
                    rows.append(f"hub: {name} {step} {branch}")
                    names.append(name)
                names.append("choke" if shared else "g")
                rows += [f"connection: {first}-{second}"
                         for first, second in zip(names, names[1:])]
            expected = 6 + 25 - 1 if shared else 6 + 25 // branches - 1
        return "\n".join(rows) + "\n", expected


ENVELOPE = ["narrow-final-cut", "parallel-pipelines", "capacity-three",
            "restricted-chain", "cyclic-bottleneck", "unusable-full-envelope"]


@pytest.mark.parametrize("family", ENVELOPE)
def test_analytic_envelope_and_planning_measurements(
    family: str, record_property: Callable[[str, object], None],
) -> None:
    """Certify larger analytic families without enumerating joint states.

    Args:
        family (str): Analytic topology family selected for lower-bound
            verification.
        record_property (Callable[[str, object], None]): Pytest report fixture
            used to attach planning measurements.
    """
    text, expected = EnvelopeMaps.build(family)
    config = Parser().parse_text(text)
    network = DroneNetwork.from_config(config)
    assert len(network.zones) <= 54 and len(network.connections) <= 70
    if family == "unusable-full-envelope":
        assert (config.nb_drones, len(network.zones),
                len(network.connections)) == (25, 54, 70)
    assert NetworkProfiler(network).profile(25).lower_bound == expected
    solver = OptimalSolver()
    simulation = Simulation.from_config(config, solver)
    tracemalloc.start()
    try:
        started = perf_counter()
        solver.prepare(simulation.network, simulation.drones)
        seconds = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    record_property("planning_seconds", seconds)
    record_property("planning_peak_bytes", peak)
    assert solver.solution is not None and solver.solution.optimal
    assert solver.solution.makespan == expected
    assert solver.solution.lower_bound == expected
    assert simulation.run(max_turns=expected + 1).turns == expected
    assert all(drone.delivered for drone in simulation.drones)


@pytest.mark.parametrize("limits,status", [
    ({"time_budget": 0.0}, "timeout"),
    ({"node_budget": 0}, "node_limit"),
])
def test_explicit_cutoff_raises_unproved_error(
    limits: dict[str, float | int], status: str,
) -> None:
    """Retain an interrupted exact search's floor as diagnostic data.

    Args:
        limits (dict[str, float | int]): Cutoff dictionary selecting a zero-
            time or zero-node proof limit.
        status (str): Expected status for the selected search cutoff.
    """
    network = DroneNetwork.from_config(Parser().parse(TRAP))
    initial = PrioritizedPlanner().solve(network, _ids(2))
    assert initial.makespan == 4
    solver = (OptimalSolver(time_budget=float(limits["time_budget"]))
              if "time_budget" in limits
              else OptimalSolver(node_budget=int(limits["node_budget"])))
    with pytest.raises(UnprovedPlanningError) as raised:
        solver.solve(network, _ids(2), initial)
    assert raised.value.status == status
    assert raised.value.incumbent is not None
    assert raised.value.incumbent.makespan == 4
    assert not raised.value.incumbent.optimal
    assert not ConflictDetector(network).find(raised.value.incumbent.paths)


def test_explicit_timeout_during_seeding_is_not_success() -> None:
    """A zero proof budget applies before constructing its first floor."""
    network = DroneNetwork.from_config(Parser().parse(TRAP))
    with pytest.raises(UnprovedPlanningError) as raised:
        OptimalSolver(time_budget=0).solve(network, _ids(2))
    assert raised.value.status == "timeout"


def test_unlimited_child_keeps_parent_deadline_and_cancellation() -> None:
    """Preserve caller interruption constraints with unlimited local limits."""
    cancel = Event()
    parent = SearchBudget(deadline=perf_counter() + 20,
                          cancelled=cancel.is_set)
    child = parent.child(None)
    assert child.deadline == parent.deadline
    cancel.set()
    with pytest.raises(PlanningCancelled):
        child.check()


@pytest.mark.parametrize("expired", ["engine", "call"])
def test_low_level_keeps_the_earlier_nested_deadline(
    expired: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new path-call budget cannot extend an existing engine deadline.

    Args:
        expired (str): Whether the engine or path call owns the already expired
            deadline.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to patch search
            behavior during the test.
    """
    network = _network("nb_drones: 1\nstart_hub: s 0 0\n"
                       "end_hub: g 1 0\nconnection: s-g\n")
    engine_deadline = perf_counter() + 60
    engine = SpaceTimeAStar(network, SearchBudget(deadline=engine_deadline))
    if expired == "engine":
        monkeypatch.setattr("fly_in.solvers.base.perf_counter",
                            lambda: engine_deadline + 1)
        call_budget = SearchBudget()
    else:
        call_budget = SearchBudget(deadline=perf_counter() - 1)
    with pytest.raises(SearchTimeout):
        engine.find_path("D1", budget=call_budget)


@pytest.mark.parametrize("cancelled", ["engine", "call"])
def test_low_level_keeps_both_nested_cancellation_tokens(
    cancelled: str,
) -> None:
    """Either caller's cancellation interrupts a nested path request.

    Args:
        cancelled (str): Whether cancellation is set on the engine or path-call
            budget.
    """
    network = _network("nb_drones: 1\nstart_hub: s 0 0\n"
                       "end_hub: g 1 0\nconnection: s-g\n")
    engine_cancel, call_cancel = Event(), Event()
    engine = SpaceTimeAStar(network, SearchBudget(
        cancelled=engine_cancel.is_set,
    ))
    (engine_cancel if cancelled == "engine" else call_cancel).set()
    with pytest.raises(PlanningCancelled):
        engine.find_path("D1", budget=SearchBudget(
            cancelled=call_cancel.is_set,
        ))


def test_expired_parent_cannot_certify_even_a_bound_matching_floor() -> None:
    """Honor an expired caller deadline before the immediate certificate."""
    network = _network("nb_drones: 1\nstart_hub: s 0 0\n"
                       "end_hub: g 1 0\nconnection: s-g\n")
    initial = PrioritizedPlanner().solve(network, _ids(1))
    for supplied in (None, initial):
        with pytest.raises(UnprovedPlanningError):
            OptimalSolver().solve(
                network, _ids(1), supplied,
                SearchBudget(deadline=perf_counter() - 1),
            )


@pytest.mark.parametrize("stage", ["seeding", "search"])
def test_cancellation_during_low_level_work_remains_cancellation(
    stage: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel inside a real low-level invocation at either planning stage.

    Args:
        stage (str): Whether cancellation is injected during floor seeding or
            CT search.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to patch search
            behavior during the test.
    """
    network = DroneNetwork.from_config(Parser().parse(TRAP))
    initial = _serial_seed(network, 2) if stage == "search" else None
    cancel = Event()
    original = SpaceTimeAStar.find_path

    def interrupted(
        engine: SpaceTimeAStar, agent: str,
        reservations: ReservationTable | None = None,
        constraints: Iterable[Constraint] = (), horizon: int = 100,
        budget: SearchBudget | None = None,
    ) -> TimedPath | None:
        """Trigger the caller token while the search invokes its low level.

        Args:
            engine (SpaceTimeAStar): Engine whose path search is intercepted.
            agent (str): Drone identifier passed to the path search.
            reservations (ReservationTable | None): Existing claims for the
                path search.
            constraints (Iterable[Constraint]): Forbidden resource uses for
                the path search.
            horizon (int): Latest arrival turn allowed by the path search.
            budget (SearchBudget | None): Cooperative limits for path search.

        Returns:
            TimedPath | None: Original path result after setting cancellation.

        Raises:
            PlanningCancelled: When the wrapped path search checks the set
                caller token.
        """
        cancel.set()
        return original(engine, agent, reservations, constraints, horizon,
                        budget)

    monkeypatch.setattr(SpaceTimeAStar, "find_path", interrupted)
    with pytest.raises(PlanningCancelled):
        OptimalSolver().solve(network, _ids(2), initial,
                              SearchBudget(cancelled=cancel.is_set))


@pytest.mark.parametrize("invalid", [
    "missing-agent", "extra-agent", "wrong-agent", "teleport", "conflicting",
])
def test_invalid_supplied_seed_never_receives_a_certificate(
    invalid: str,
) -> None:
    """Reject illegal floors despite bound equality or initial proof flags.

    Args:
        invalid (str): Malformed incumbent case selected by pytest.
    """
    network = DroneNetwork.from_config(Parser().parse(TRAP))
    initial = PrioritizedPlanner().solve(network, _ids(2))
    paths = dict(initial.paths)
    if invalid == "missing-agent":
        del paths["D2"]
    elif invalid == "extra-agent":
        paths["D3"] = replace(paths["D1"], agent="D3")
    elif invalid == "wrong-agent":
        paths["D1"] = replace(paths["D1"], agent="D2")
    elif invalid == "teleport":
        paths["D1"] = TimedPath("D1", (("s", 0), ("g", 3)), ())
    else:
        engine = SpaceTimeAStar(network)
        for agent in _ids(2):
            path = engine.find_path(agent)
            assert path is not None
            paths[agent] = path
    seed = Solution(paths, optimal=True, status="optimal", reason="untrusted")
    with pytest.raises(ValueError):
        OptimalSolver().solve(network, _ids(2), seed)


def test_unreachable_map_has_a_controlled_error() -> None:
    """No proof or feasible floor is fabricated for a disconnected goal."""
    network = _network("nb_drones: 1\nstart_hub: s 0 0\n"
                       "end_hub: g 1 0\nhub: a 0 1\nconnection: s-a\n")
    with pytest.raises(ValueError):
        OptimalSolver().solve(network, _ids(1))


@pytest.mark.parametrize("limits", [{"time_budget": 0.0}, {"node_budget": 0}])
def test_gui_unproved_work_emits_failure_without_prepared(
    limits: dict[str, float | int], app: QApplication, tmp_path: Path,
) -> None:
    """Actual proof-mode worker cutoffs must never transfer playback state.

    Args:
        limits (dict[str, float | int]): Cutoff dictionary selecting a zero-
            time or zero-node proof limit.
        app (QApplication): Qt application fixture used to process planning-
            worker signals.
        tmp_path (Path): Pytest temporary directory used for logs or generated
            input.
    """
    worker = PlanningWorker(19, Parser().parse(TRAP), "optimal", limits,
                            records_dir=tmp_path)
    prepared: list[tuple[object, ...]] = []
    failures: list[tuple[int, str]] = []
    worker.prepared.connect(lambda *args: prepared.append(args))
    worker.failed.connect(lambda generation, message:
                          failures.append((generation, message)))
    worker.run()
    app.processEvents()
    assert prepared == []
    assert len(failures) == 1 and failures[0][0] == 19
    assert "prov" in failures[0][1].lower()
