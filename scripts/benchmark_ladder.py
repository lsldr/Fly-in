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
    """Compare configured strategies while preserving report provenance."""

    def __init__(
        self, map_root: Path, records_root: Path, time_budget: float,
        export_markdown: bool, strategies: tuple[str, ...] = STRATEGIES,
    ) -> None:
        """Capture catalog, destinations, limits, and measured solver order."""
        self.map_root = map_root
        self.records_root = records_root
        self.time_budget = time_budget
        self.export_markdown = export_markdown
        self.strategies = strategies

    def _cell(self, row: dict[str, object]) -> str:
        """Format one measured result without claiming unproved optimality."""
        if row.get('status') != 'completed':
            return 'failed'
        value = f"{row['turns']} / {row['total_path_cost']}"
        if row.get('optimal_makespan') is True:
            value += ' (optimal)'
        elif row.get('search_status') in ('timeout', 'node_limit'):
            value += f" ({row['search_status']})"
        return value

    def run(self) -> tuple[dict[str, object], Path, Path | None]:
        """Run selected strategies (all six by default) and render results."""
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
            '| Map | Drones | Target | ' + ' | '.join(labels) + ' | Winner |',
            '|---|---:|---:|' + '---:|' * len(labels) + '---|',
        ]
        summary = []
        for name, by_strategy in sorted(comparisons.items()):
            first = by_strategy[strategies[0]]
            target = first.get('target_turns')
            target_text = (
                ('<' if name.startswith('challenger/') else '≤') + str(target)
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
    """Own comparative benchmark argument handling and execution."""

    def run(self) -> None:
        """Parse the comparative benchmark command."""
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
        BenchmarkLadder(
            arguments.map_root, arguments.records_root,
            arguments.time_budget, arguments.export_markdown,
            (arguments.solver,) if arguments.solver else STRATEGIES,
        ).run()


def main() -> None:
    """Delegate the ladder command to its application owner."""
    LadderApplication().run()


if __name__ == '__main__':
    main()
