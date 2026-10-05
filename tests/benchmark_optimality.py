"""Repeatable catalog/envelope benchmark with independent output validation.

Example: .venv/bin/python tests/benchmark_optimality.py --repeats 3
Optional --report /tmp/fly-in-review.json writes detailed measured evidence.
No production artifact directories or documentation are modified.
Planning is timed separately from execution. Peak traced Python memory is
measured in a separate solve, so tracing overhead does not distort timings.
"""

import argparse
import json
from math import isfinite
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
import tracemalloc

from joint_oracle import exact_makespan
from optimality_cases import (
    CATALOG_TURNS, EXTRA, HARD_CASE, LARGE_CASES, ROOT, SMALL_NAMES,
)
from schedule_replay import replay
from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.simulation import Simulation
from fly_in.solver_factory import DEFAULT_SOLVER, SolverFactory
from fly_in.solvers.optimal import OptimalSolver, UnprovedPlanningError


def benchmark(path: Path, minimum: int, repeats: int) -> dict[str, object]:
    """Measure fresh default preparations, execution, and separate memory."""
    config = Parser().parse(path)
    seconds = []
    execution = []
    for _ in range(repeats):
        solver = SolverFactory().create(DEFAULT_SOLVER)
        assert isinstance(solver, OptimalSolver)
        simulation = Simulation.from_config(config, solver)
        started = perf_counter()
        solver.prepare(simulation.network, simulation.drones)
        seconds.append(perf_counter() - started)
        started = perf_counter()
        result = simulation.run(max_turns=minimum + 1)
        execution.append(perf_counter() - started)
        assert result.turns == minimum
        assert solver.solution is not None and solver.solution.optimal
        assert replay(config, result.history).cost == solver.solution.cost
    tracemalloc.start()
    try:
        measured = SolverFactory().create(DEFAULT_SOLVER)
        assert isinstance(measured, OptimalSolver)
        memory_simulation = Simulation.from_config(config, measured)
        measured.prepare(memory_simulation.network, memory_simulation.drones)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert isinstance(solver, OptimalSolver)
    solution = solver.solution
    assert solution is not None
    return {
        "map": path.relative_to(ROOT / "maps").as_posix(),
        "drones": config.nb_drones, "hubs": len(config.zones) + 2,
        "connections": len(config.connections),
        "minimum_turns": minimum, "achieved_turns": result.turns,
        "gap": result.turns - minimum, "optimal": solution.optimal,
        "flow_bound": solution.lower_bound, "proof": solution.reason,
        "ct_nodes": solution.stats.nodes,
        "planning_seconds": seconds, "planning_median": median(seconds),
        "execution_seconds": execution, "peak_python_bytes": peak,
    }


def benchmark_crossing(
    repeats: int, proof_seconds: float,
) -> dict[str, object]:
    """Bound a deliberately difficult exact search and report honest gaps."""
    config = Parser().parse(EXTRA / HARD_CASE.filename)
    measurements = []
    for _ in range(repeats):
        solver = OptimalSolver(time_budget=proof_seconds)
        simulation = Simulation.from_config(config, solver)
        started = perf_counter()
        try:
            solver.prepare(simulation.network, simulation.drones)
        except UnprovedPlanningError as error:
            solution = error.incumbent
            assert solution is not None and not solution.optimal
            status = error.status
        else:
            solution = solver.solution
            assert solution is not None and solution.optimal
            assert solution.makespan == HARD_CASE.minimum()
            status = "optimal"
        measurements.append({
            "seconds": perf_counter() - started, "status": status,
            "diagnostic_turns": solution.makespan,
            "diagnostic_gap": solution.makespan - HARD_CASE.minimum(),
            "flow_bound": solution.lower_bound,
            "ct_nodes": solution.stats.nodes, "optimal": solution.optimal,
        })
    return {
        "map": f"extra/{HARD_CASE.filename}", "drones": config.nb_drones,
        "hubs": len(config.zones) + 2, "connections": len(config.connections),
        "minimum_turns": HARD_CASE.minimum(), "proof": HARD_CASE.proof,
        "time_budget": proof_seconds, "measurements": measurements,
    }


def main() -> None:
    """Benchmark all reviewed fixtures and display concrete results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--difficult-budget", type=float, default=5.0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not isfinite(args.difficult_budget) or args.difficult_budget <= 0:
        parser.error("--difficult-budget must be finite and positive")
    cases = [(ROOT / "maps" / name, minimum)
             for name, minimum in CATALOG_TURNS.items()]
    cases += [(EXTRA / case.filename, case.minimum()) for case in LARGE_CASES]
    for name in SMALL_NAMES:
        config = Parser().parse(EXTRA / name)
        cases.append((EXTRA / name, exact_makespan(
            DroneNetwork.from_config(config), config.nb_drones,
        )))
    results = []
    for path, minimum in cases:
        row = benchmark(path, minimum, args.repeats)
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    difficult = benchmark_crossing(args.repeats, args.difficult_budget)
    results.append(difficult)
    print(json.dumps(difficult, sort_keys=True), flush=True)
    if args.report is not None:
        args.report.write_text(json.dumps({
            "solver": DEFAULT_SOLVER, "repeats": args.repeats,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "results": results,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
