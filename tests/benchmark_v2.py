"""Audit every catalog map and solver with independent replay and a watchdog.

Run: .venv/bin/python tests/benchmark_v2.py --report /tmp/fly-in-v2.json
Strategies retain their actual factory defaults. An external process watchdog
detects a planning/turn/validation stage with no completion for 120 seconds;
total duration may exceed that while stages continue completing. Failed rows
are saved immediately and produce a nonzero exit. Expected unreachable maps
are classified separately. Timings are evidence, not optimality certificates.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter
import traceback

from fly_in.parser import Parser
from fly_in.simulation import Simulation
from fly_in.solver_factory import SOLVER_CHOICES, SolverFactory
from fly_in.solvers.base import ScheduledSolver
from fly_in.solvers.protocol import PreparableSolver
from joint_oracle import exact_makespan
from optimality_cases import (
    CATALOG_TURNS, HARD_CASE, LARGE_CASES, ROOT, SMALL_NAMES,
)
from fly_in.realm.network import DroneNetwork
from schedule_replay import replay


@dataclass
class Outcome:
    """Validated results or a concrete failure for one map/strategy pair."""

    map: str
    solver: str
    expected_minimum: int | None
    status: str = "failed"
    turns: int | None = None
    cost: int | None = None
    optimal: bool | None = None
    proof_status: str | None = None
    proof_reason: str | None = None
    ct_nodes: int | None = None
    bypasses: int | None = None
    planning_seconds: float = 0
    total_seconds: float = 0
    error: str | None = None


def run_worker(
    path: Path, strategy: str, minimum: int | None,
    unsolvable: bool, pipe: Connection,
) -> None:
    """Execute real defaults and report each completed simulation stage."""
    row = Outcome(path.relative_to(ROOT / "maps").as_posix(), strategy,
                  minimum)
    started = perf_counter()
    try:
        with TemporaryDirectory(prefix="fly-in-v2-") as artifacts:
            pipe.send("parse")
            config = Parser().parse(path)
            solver = SolverFactory(path.stem, Path(artifacts)).create(strategy)
            simulation = Simulation.from_config(config, solver)
            pipe.send("planning")
            planning_start = perf_counter()
            if isinstance(solver, PreparableSolver):
                solver.prepare(simulation.network, simulation.drones)
            row.planning_seconds = perf_counter() - planning_start
            pipe.send("turn 1")

            def completed_turn(current: Simulation, moves: list[str]) -> None:
                """Reset the parent watchdog after each completed turn."""
                pipe.send(f"turn {current.turn + 1}")

            result = simulation.run(on_turn=completed_turn)
            row.turns = result.turns
            row.cost = result.metrics.total_path_cost
            pipe.send("independent replay")
            assert not unsolvable, "a disconnected map was delivered"
            independent = replay(config, result.history)
            assert independent.cost == row.cost
            if isinstance(solver, ScheduledSolver):
                solution = solver.solution
                assert solution is not None
                assert solution.makespan == result.turns
                row.optimal = solution.optimal
                row.proof_status = solution.status
                row.proof_reason = solution.reason
                row.ct_nodes = solution.stats.nodes
                row.bypasses = solution.stats.bypasses
                if solution.optimal:
                    assert solution.status == "optimal" and solution.reason
                    assert minimum is not None and result.turns == minimum
            if strategy == "optimal":
                assert row.optimal and result.turns == minimum
            row.status = "completed"
    except ValueError as error:
        row.error = str(error)
        if unsolvable and ("unreachable" in row.error or
                           "no accessible route" in row.error):
            row.status = "expected_unsolvable"
        else:
            row.error = traceback.format_exc()
    except Exception:
        row.error = traceback.format_exc()
    finally:
        row.total_seconds = perf_counter() - started
        pipe.send(row)
        pipe.close()


def run_case(
    path: Path, strategy: str, minimum: int | None, unsolvable: bool,
) -> Outcome:
    """Isolate hard hangs, including code that misses cooperative checks."""
    context = mp.get_context("spawn")
    incoming, outgoing = context.Pipe(duplex=False)
    process = context.Process(target=run_worker,
                              args=(path, strategy, minimum,
                                    unsolvable, outgoing))
    row = Outcome(path.relative_to(ROOT / "maps").as_posix(), strategy,
                  minimum)
    started = last_progress = monotonic()
    stage = "worker startup"
    process.start()
    outgoing.close()
    try:
        while True:
            if incoming.poll(1):
                try:
                    message = incoming.recv()
                except EOFError:
                    row.error = f"worker closed during {stage}"
                    return row
                if isinstance(message, Outcome):
                    return message
                assert isinstance(message, str)
                stage = message
                last_progress = monotonic()
            if monotonic() - last_progress > 120:
                row.status = "stalled"
                row.error = f"No completed stage for 120 seconds: {stage}"
                row.total_seconds = monotonic() - started
                return row
            if not process.is_alive() and not incoming.poll():
                row.error = f"worker exited {process.exitcode} during {stage}"
                return row
    finally:
        incoming.close()
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def main() -> None:
    """Persist all cases, known minima, expected errors and any failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=Path("/tmp/fly-in-v2-benchmark.json"))
    args = parser.parse_args()
    minima = dict(CATALOG_TURNS)
    for case in (*LARGE_CASES, HARD_CASE):
        minima[f"extra/{case.filename}"] = case.minimum()
    for name in (*[f"extra/{name}" for name in SMALL_NAMES],
                 "unsolvable/06_reachable_control.txt"):
        config = Parser().parse(ROOT / "maps" / name)
        minima[name] = exact_makespan(
            DroneNetwork.from_config(config), config.nb_drones,
        )
    rows = []
    failures = []
    for strategy in SOLVER_CHOICES:
        for path in sorted((ROOT / "maps").rglob("*.txt")):
            relative = path.relative_to(ROOT / "maps").as_posix()
            unsolvable = (path.parent.name == "unsolvable" and
                          not path.name.startswith("06_"))
            assert unsolvable or relative in minima, relative
            print(f"START {strategy} {relative}", flush=True)
            row = run_case(path, strategy, minima.get(relative), unsolvable)
            rows.append(asdict(row))
            if row.status not in ("completed", "expected_unsolvable"):
                failures.append(asdict(row))
            print(f"{row.status} {strategy} {relative}: turns={row.turns}",
                  flush=True)
            args.report.write_text(json.dumps({
                "date": "2026-10-13", "python": platform.python_version(),
                "platform": platform.platform(),
                "stage_watchdog_seconds": 120, "results": rows,
                "failures": failures,
            }, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} cases, {len(failures)} failures", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
