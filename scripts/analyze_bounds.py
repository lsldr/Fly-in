"""Analyze shortest latency, bottleneck capacity, and makespan bounds."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fly_in.errors import ConfigParseError
from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.solvers.bounds import NetworkProfiler


class BoundAnalysisApplication:
    """A command-line application for necessary delivery-time bounds.

    Use the JSON output to inspect the fields of
    fly_in.solvers.bounds.InstanceProfile. A matching feasible schedule is
    still required to prove that a lower bound is attainable.

    Attributes:
        argument_parser (argparse.ArgumentParser): Map path and output options.
        map_parser (Parser): Reader for validated map declarations.
    """

    def __init__(self) -> None:
        """Prepare the command's argument parser and map reader."""
        self.argument_parser = argparse.ArgumentParser(description=__doc__)
        self.argument_parser.add_argument(
            "map", type=Path, help="path to a map .txt file",
        )
        self.argument_parser.add_argument(
            "--json", action="store_true", help="print JSON",
        )
        self.map_parser = Parser()

    def run(self) -> None:
        """Analyze the map and print its delivery-time lower bound.

        Read arguments from sys.argv. Print a human-readable report or JSON to
        standard output; report parsing and unreachable-map errors on stderr.

        Raises:
            SystemExit: Help was requested (code 0), or arguments, map
                contents,
                or network reachability are invalid (code 2).
        """
        args = self.argument_parser.parse_args()
        try:
            config = self.map_parser.parse(args.map)
            profile = NetworkProfiler(
                DroneNetwork.from_config(config),
            ).profile(config.nb_drones)
        except (ConfigParseError, ValueError) as error:
            self.argument_parser.exit(2, f"error: {error}\n")
        if args.json:
            print(json.dumps(
                {"map": str(args.map), **asdict(profile)}, indent=2,
            ))
            return
        print(f"Map: {args.map}")
        print(f"Fleet: {profile.fleet_size} drones")
        print(f"Graph: {profile.vertices} hubs, {profile.edges} connections")
        print(f"Density (E/V): {profile.density:.3f}")
        print(f"Restricted hubs: {profile.restricted_ratio:.1%}")
        print(f"Shortest latency: {profile.shortest_distance} turns")
        print(
            f"Shared-resource min-cut: {profile.bottleneck_capacity} per turn"
        )
        print(f"Makespan lower bound: {profile.lower_bound} turns")
        print("A valid schedule matching this bound has optimal makespan.")
        print("The bound alone does not construct a feasible schedule.")


def main() -> None:
    """Run the map-bound analysis application using sys.argv.

    Raises:
        SystemExit: Help was requested (code 0) or analysis failed (code 2).
    """
    BoundAnalysisApplication().run()


if __name__ == "__main__":
    main()
