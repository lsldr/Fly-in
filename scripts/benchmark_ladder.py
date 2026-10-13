"""Compare every solver rung on the supplied map library.

Usage: uv run python scripts/benchmark_ladder.py --time-budget 1.0
       uv run python scripts/benchmark_ladder.py --solver adaptive
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import cast

from fly_in.benchmark import BenchmarkRunner
from fly_in.output import AtomicFileWriter
from fly_in.solver_factory import SOLVER_CHOICES

STRATEGIES = ('priority', 'greedy', 'lns', 'pbs', 'icbs', 'adaptive')
LABELS = {
    'optimal': 'Optimal (proof required)',
    'priority': 'Rung 0',
    'greedy': 'Greedy',
    'lns': 'Rung 1 LNS',
    'pbs': 'Rung 2 PBS',
    'icbs': 'Rung 3 ICBS',
    'adaptive': 'Adaptive',
}


class BenchmarkLadder:
    """A comparison of selected solvers over a map catalog.

    Attributes:
        map_root (Path): Root of the map catalog searched recursively.
        records_root (Path): Destination for component and combined reports.
        time_budget (float): Planning budget in seconds passed to each solver.
        export_markdown (bool): Whether to save the printed comparison table.
        strategies (tuple[str, ...]): Solver order, also used to break ties.
    """

    def __init__(
        self, map_root: Path, records_root: Path, time_budget: float,
        export_markdown: bool, strategies: tuple[str, ...] = STRATEGIES,
    ) -> None:
        """Capture the catalog, report settings, and solver order.

        Args:
            map_root (Path): Map catalog directory.
            records_root (Path): Destination for benchmark and solver records.
            time_budget (float): Planning budget in seconds for each solver.
            export_markdown (bool): Whether to save a Markdown comparison
                table.
            strategies (tuple[str, ...]): Solver names in comparison order;
                defaults to the six heuristic strategies in STRATEGIES.
        """
        self.map_root = map_root
        self.records_root = records_root
        self.time_budget = time_budget
        self.export_markdown = export_markdown
        self.strategies = strategies

    def _cell(self, row: dict[str, object]) -> str:
        """Format a result with its measured cost and proof or cutoff status.

        Args:
            row (dict[str, object]): Result row from BenchmarkRunner.

        Returns:
            str: Turn count and physical path cost, with an optimality or
                cutoff
            suffix when present; "failed" for an incomplete run.
        """
        if row.get('status') != 'completed':
            if row.get('optimal_makespan') is True:
                return 'failed (plan proved)'
            if row.get('search_status') in ('timeout', 'node_limit'):
                return f"failed ({row['search_status']}; unproved)"
            return 'failed'
        value = f"{row['turns']} / {row['total_path_cost']}"
        if row.get('optimal_makespan') is True:
            value += ' (optimal)'
        elif row.get('search_status') in ('timeout', 'node_limit'):
            value += f" (unproved; {row['search_status']})"
        else:
            value += ' (unproved)'
        if row.get('meets_target') is False:
            value += ' (target missed)'
        return value

    def run(self) -> tuple[dict[str, object], Path, Path | None]:
        """Benchmark selected strategies and persist the comparison reports.

        Print the Markdown table and report paths to stdout. Select each winner
        by turn count, then physical path cost, then the configured solver
        order.
        Individual map failures remain in the component reports.

        Returns:
            tuple[dict[str, object], Path, Path | None]: Combined JSON-
                compatible
            report, saved JSON path, and Markdown path, or None if export is
            off.

        Raises:
            ValueError: No strategy was selected, a solver name is unsupported,
                or the catalog contains no text maps.
            OSError: A component or combined report cannot be persisted.
        """
        strategies = self.strategies
        if not strategies or any(
            name not in SOLVER_CHOICES for name in strategies
        ):
            raise ValueError('select one or more supported solver strategies')
        comparisons: dict[str, dict[str, dict[str, object]]] = {}
        component_reports: dict[str, str] = {}
        for strategy in strategies:
            report, path = BenchmarkRunner(
                self.map_root, strategy, self.records_root,
                {'time_budget': self.time_budget},
            ).run()
            component_reports[strategy] = str(path)
            for item in cast(list[dict[str, object]], report['results']):
                comparisons.setdefault(str(item['map']), {})[strategy] = item
        labels = [LABELS[name] for name in strategies]
        lines = [
            '| Map | Drones | Target | ' + ' | '.join(labels)
            + ' | Best observed |',
            '|---|---:|---:|' + '---:|' * len(labels) + '---|',
        ]
        summary = []
        for name, by_strategy in sorted(comparisons.items()):
            first = by_strategy[strategies[0]]
            target = first.get('target_turns')
            target_text = (
                ('<' if (name.startswith('challenger/')
                         or self.map_root.name == 'challenger') else '≤')
                + str(target)
                if target is not None else '—'
            )
            successful = [
                (cast(int, row['turns']),
                 cast(int, row['total_path_cost']), index, strategy)
                for index, strategy in enumerate(strategies)
                for row in (by_strategy[strategy],)
                if row.get('status') == 'completed'
            ]
            winner = min(successful)[3] if successful else 'none'
            row = {
                'map': name,
                'target_turns': first.get('target_turns'),
                'winner': winner,
                'strategies': by_strategy,
            }
            summary.append(row)
            lines.append('| ' + ' | '.join([
                name,
                str(first.get('nb_drones', '—')),
                target_text,
                *(self._cell(by_strategy[key]) for key in strategies),
                winner,
            ]) + ' |')
        timestamp = datetime.now(timezone.utc)
        stamp = timestamp.strftime('%Y%m%dT%H%M%S%fZ')
        combined: dict[str, object] = {
            'schema_version': 1,
            'timestamp': timestamp.isoformat(),
            'time_budget_seconds': self.time_budget,
            'solvers': list(strategies),
            'map_root': str(self.map_root.resolve()),
            'environment': {
                'python': platform.python_version(),
                'platform': platform.platform(),
            },
            'component_reports': component_reports,
            'passed': all(
                row.get('status') == 'completed'
                and row.get('meets_target') is not False
                for by_strategy in comparisons.values()
                for row in by_strategy.values()
            ),
            'results': summary,
        }
        report_path = self.records_root / f'ladder_{stamp}.json'
        AtomicFileWriter(report_path).write(
            json.dumps(combined, indent=2) + '\n'
        )
        markdown_path = None
        if self.export_markdown:
            markdown_path = self.records_root / f'ladder_{stamp}.md'
            AtomicFileWriter(markdown_path).write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        print(f'JSON report: {report_path}')
        if markdown_path is not None:
            print(f'Markdown table: {markdown_path}')
        return combined, report_path, markdown_path


class LadderApplication:
    """Command-line options for comparing catalog benchmark results.
    """

    def run(self) -> int:
        """Parse sys.argv, run catalog benchmarks, and print the comparison.

        Returns:
            int: Zero if every run completes and meets its known target;
                one if any run fails or misses its target.

        Raises:
            SystemExit: Help was requested (code 0) or arguments are invalid
                (code 2), including a negative time budget.
            ValueError: The benchmark configuration is invalid.
            OSError: A benchmark report cannot be persisted.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--map-root', type=Path, default=Path('maps'))
        parser.add_argument('--records-root', type=Path,
                            default=Path('records'))
        parser.add_argument('--time-budget', type=float, default=1.0)
        parser.add_argument('--export-markdown', action='store_true')
        parser.add_argument('--solver', choices=SOLVER_CHOICES,
                            help='run only this solver (default: run all six)')
        arguments = parser.parse_args()
        if arguments.time_budget < 0:
            parser.error('--time-budget must be nonnegative')
        report, _, _ = BenchmarkLadder(
            arguments.map_root, arguments.records_root,
            arguments.time_budget, arguments.export_markdown,
            (arguments.solver,) if arguments.solver else STRATEGIES,
        ).run()
        return 0 if report['passed'] else 1


def main() -> None:
    """Run the comparative benchmark application using sys.argv.

    Raises:
        SystemExit: Zero on success, one on failed runs or missed targets,
            or two for invalid command-line arguments.
        ValueError: The benchmark configuration is invalid.
        OSError: A benchmark report cannot be persisted.
    """
    raise SystemExit(LadderApplication().run())


if __name__ == '__main__':
    main()
