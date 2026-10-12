"""Command-line selection, validation, and simulation entrypoint."""

import argparse
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal

from fly_in.errors import ConfigParseError
from fly_in.models import MapConfig
from fly_in.parser import Parser
from fly_in.provenance import PlanningProvenance
from fly_in.simulation import Simulation
from fly_in.solvers import PreparableSolver
from fly_in.solver_factory import (
    DEFAULT_SOLVER,
    SOLVER_CHOICES,
    SolverFactory,
)
from fly_in.visualization.terminal import TerminalVisualizer


class CliApplication:
    """Command-line interface and lifecycle manager for selected runs.

    Attributes:
        parser (Parser): Map parser used for validation and execution.
    """

    def __init__(self, parser: Parser | None = None) -> None:
        """Bind the map parser used for validation and terminal execution.

        Args:
            parser (Parser | None): Parser to use, or ``None`` to create the
                default parser.
        """
        self.parser = parser or Parser()

    def select_map(self, map_root: Path | None = None) -> Path:
        """Prompt for a map from a sorted recursive directory listing.

        Args:
            map_root (Path | None): Search root, defaulting to ``./maps`` in
                the current working directory.

        Returns:
            Path: Selected map path.

        Raises:
            FileNotFoundError: If the search root contains no text maps.
            RuntimeError: If input ends before the user selects a map.
        """
        map_root = map_root or Path.cwd() / "maps"
        maps = sorted(
            (
                path
                for path in map_root.rglob("*")
                if path.is_file() and path.suffix.lower() == ".txt"
            ),
            key=lambda path: path.relative_to(map_root).as_posix().lower(),
        )
        if not maps:
            raise FileNotFoundError(
                f"No .txt files found under '{map_root.resolve()}'"
            )
        print("Available maps:", file=sys.stderr)
        for index, map_file in enumerate(maps, start=1):
            print(
                f"{index}. {map_file.relative_to(map_root)}",
                file=sys.stderr,
            )
        while True:
            choice = self._read_choice("Select a map: ")
            try:
                index = int(choice) - 1
                if 0 <= index < len(maps):
                    return maps[index]
            except ValueError:
                pass
            print("Invalid selection. Please try again.", file=sys.stderr)

    def _read_choice(self, prompt: str) -> str:
        """Read a prompt from stdin while writing the prompt to stderr.

        Args:
            prompt (str): Prompt text displayed before reading input.

        Returns:
            str: User-entered choice.

        Raises:
            RuntimeError: If stdin reaches EOF before a choice is entered.
        """
        print(prompt, end="", file=sys.stderr, flush=True)
        try:
            return input()
        except EOFError as error:
            raise RuntimeError("selection ended before a choice") from error

    def select_mode(self) -> bool:
        """Prompt until the user selects terminal or desktop mode.

        Returns:
            bool: ``True`` for GUI mode and ``False`` for terminal mode.

        Raises:
            RuntimeError: If input ends before a mode is selected.
        """
        print("1. Terminal\n2. GUI", file=sys.stderr)
        while True:
            choice = self._read_choice("Select a mode: ")
            if choice in {"1", "2"}:
                return choice == "2"
            print("Invalid selection. Please try again.", file=sys.stderr)

    def launch_gui(
        self,
        path: Path,
        map_root: Path,
        solver_name: str,
        output_dir: Path,
        records_dir: Path,
        solver_parameters: dict[str, float | int] | None = None,
    ) -> None:
        """Launch desktop playback, importing Qt only when requested.

        Args:
            path (Path): Initial map to load.
            map_root (Path): Root directory used by map selection.
            solver_name (str): Strategy selected for planning.
            output_dir (Path): Destination for successful movement output.
            records_dir (Path): Destination for run records.
            solver_parameters (dict[str, float | int] | None): Optional
                planning limits and strategy settings.
        """
        from fly_in.gui.main_window import DesktopApplication

        DesktopApplication(
            initial_map=path,
            map_root=map_root,
            solver_name=solver_name,
            output_dir=output_dir,
            records_dir=records_dir,
            solver_parameters=solver_parameters,
            argv=[sys.argv[0]],
        ).run()

    def argument_parser(self) -> argparse.ArgumentParser:
        """Define CLI flags while preserving direct map invocation.

        Returns:
            argparse.ArgumentParser: Parser configured for the fly-in CLI.
        """
        parser = argparse.ArgumentParser(
            prog="fly-in",
            description="Simulate drones on a map",
        )
        parser.add_argument("config", nargs="?", help="path to a map")
        parser.add_argument(
            "--validate",
            action="store_true",
            help="validate without simulating",
        )
        parser.add_argument(
            "--no-color", action="store_true", help="disable terminal colors"
        )
        parser.add_argument(
            "--solver",
            choices=SOLVER_CHOICES,
            default=DEFAULT_SOLVER,
            help="routing strategy",
        )
        modes = parser.add_mutually_exclusive_group()
        modes.add_argument(
            "--gui",
            action="store_true",
            help="desktop playback",
        )
        modes.add_argument(
            "--terminal", action="store_true", help="terminal simulation"
        )
        parser.add_argument(
            "--map-root",
            type=Path,
            default=Path.cwd() / "maps",
            help="map library directory (default: ./maps)",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path.cwd() / "out",
            help="movement output directory (default: ./out)",
        )
        parser.add_argument(
            "--records-dir",
            type=Path,
            default=Path.cwd() / "records",
            help="JSON record directory (default: ./records)",
        )
        parser.add_argument(
            "--display",
            choices=("visual", "compact", "full", "moves"),
            default="visual",
            help="terminal detail level",
        )
        parser.add_argument(
            "--time-budget",
            type=float,
            default=None,
            help="planning seconds; unlimited optimal, heuristic default 1",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="deterministic LNS random seed (default: 0)",
        )
        parser.add_argument(
            "--max-iter",
            type=int,
            default=None,
            help="maximum LNS repairs (default: 200)",
        )
        parser.add_argument(
            "--node-budget",
            type=int,
            default=None,
            help="CT nodes; unlimited optimal, heuristic default 10000",
        )
        return parser

    def run(self, argv: list[str] | None = None) -> None:
        """Execute validation, playback, or benchmarking from arguments.

        Args:
            argv (list[str] | None): Arguments excluding the executable name;
                ``None`` uses the process command line.

        Raises:
            SystemExit: For argparse usage errors, benchmark exit status, or
                invalid run configuration.
        """
        arguments_list = list(sys.argv[1:] if argv is None else argv)
        if arguments_list and arguments_list[0] == "benchmark":
            from fly_in.benchmark import BenchmarkApplication

            raise SystemExit(BenchmarkApplication().run(arguments_list[1:]))
        parser = self.argument_parser()
        arguments = parser.parse_args(arguments_list)
        parameters = {
            key: value
            for key, value in (
                ("time_budget", arguments.time_budget),
                ("seed", arguments.seed),
                ("max_iter", arguments.max_iter),
                ("node_budget", arguments.node_budget),
            )
            if value is not None
        }
        arguments.solver_parameters = parameters
        if not arguments.config and not sys.stdin.isatty():
            parser.error("provide a map path when stdin is not interactive")
        try:
            path = (
                Path(arguments.config).expanduser()
                if arguments.config
                else self.select_map(arguments.map_root.expanduser())
            )
            config = self.parser.parse(path)
            if arguments.validate:
                print(f"{path} is a valid map configuration.")
                return
            use_gui = arguments.gui
            if (
                not arguments.config
                and not arguments.gui
                and not arguments.terminal
            ):
                use_gui = self.select_mode()
            if use_gui:
                gui_args = (
                    path,
                    arguments.map_root.expanduser(),
                    arguments.solver,
                    arguments.output_dir.expanduser(),
                    arguments.records_dir.expanduser(),
                )
                if parameters:
                    self.launch_gui(*gui_args, solver_parameters=parameters)
                else:
                    self.launch_gui(*gui_args)
                return
            self._run_terminal(path, config, arguments)
        except (ConfigParseError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))

    def _run_terminal(
        self,
        path: Path,
        config: "MapConfig",
        arguments: argparse.Namespace,
    ) -> None:
        """Execute and persist a run, including partial failure records.

        Args:
            path (Path): Source map path.
            config (MapConfig): Parsed map configuration.
            arguments (argparse.Namespace): Parsed CLI options and paths.

        Raises:
            RuntimeError: If planning or simulation fails, including when a
                keyboard interruption is converted to cancellation.
            ValueError: If solver actions violate simulation constraints.
            OSError: If source or run artifacts cannot be accessed.
        """
        from fly_in.runs import RunSession

        simulation = Simulation.from_config(
            config,
            SolverFactory(
                path.stem,
                arguments.records_dir.expanduser(),
            ).create(arguments.solver, arguments.solver_parameters),
        )
        session = RunSession(
            path,
            arguments.solver,
            map_root=arguments.map_root.expanduser(),
            output_root=arguments.output_dir.expanduser(),
            records_root=arguments.records_dir.expanduser(),
            solver_parameters=arguments.solver_parameters,
        )
        visualizer = TerminalVisualizer(
            config,
            no_color=arguments.no_color,
            display=arguments.display,
        )
        try:
            started = perf_counter()
            if isinstance(simulation.solver, PreparableSolver):
                simulation.solver.prepare(
                    simulation.network,
                    simulation.drones,
                )
            session.planning_duration_seconds = perf_counter() - started
            session.planning_provenance = PlanningProvenance.capture(
                simulation.solver,
            )
            if arguments.display == "visual":
                visualizer.render_turn(simulation, [])
            result = simulation.run(on_turn=visualizer.render_turn)
        except (RuntimeError, ValueError, KeyboardInterrupt) as error:
            status: Literal["cancelled", "failed"] = (
                "cancelled"
                if isinstance(error, KeyboardInterrupt)
                else "failed"
            )
            if session.planning_provenance is None:
                session.planning_duration_seconds = perf_counter() - started
                session.planning_provenance = PlanningProvenance.capture(
                    simulation.solver,
                    error if isinstance(error, Exception) else None,
                )
            session.finalize(simulation, status=status, error=str(error))
            if isinstance(error, KeyboardInterrupt):
                raise RuntimeError("simulation cancelled") from error
            raise
        artifacts = session.finalize(simulation)
        print(
            f"Completed with {arguments.solver} solver "
            f"in {result.turns} turns; "
            f"{result.metrics.total_moves} drone moves; "
            f"average {result.metrics.average_moves_per_turn:.2f} "
            "moves/turn.\n"
            f"Movement output saved to {artifacts.output_path}\n"
            f"Record saved to {artifacts.record_path}",
            file=sys.stderr,
        )
