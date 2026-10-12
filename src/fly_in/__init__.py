"""Public configuration parsing interface for the fly-in package.

Use Parser to load a fly_in.models.MapConfig. To execute a map, create a
strategy with fly_in.solver_factory.SolverFactory and pass the configuration
and solver to fly_in.simulation.Simulation.from_config. Terminal and desktop
clients share fly_in.runs.RunSession for output and provenance records.
"""

from fly_in.errors import ConfigParseError
from fly_in.parser import Parser

__all__ = ["ConfigParseError", "Parser"]
