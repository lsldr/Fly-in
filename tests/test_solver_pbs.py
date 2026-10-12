"""Priority DAG and PBS capacity-search behavior."""

from fly_in.parser import Parser
from fly_in.simulation import Simulation
from fly_in.solvers.base import SearchBudget, SpaceTimeAStar, ConflictDetector
from fly_in.solvers.pbs import PBSSolver, PriorityDAG


def _network(drones: int, capacity: int) -> Simulation:
    """Build a simulation with one capacity-limited intermediate hub.

    Args:
        drones (int): Fleet size used to build and solve the generated
            topology.
        capacity (int): Capacity shared by generated intermediate hubs and
            connections.

    Returns:
        Simulation: Initialized simulation using a bounded PBS solver.
    """
    text = f'''nb_drones: {drones}
start_hub: s 0 0
end_hub: g 2 0
hub: h 1 0 [max_drones={capacity}]
connection: s-h [max_link_capacity={capacity}]
connection: h-g [max_link_capacity={capacity}]
'''
    return Simulation.from_config(
        Parser().parse_text(text), PBSSolver(time_budget=0.5),
    )


def test_tarjan_cycle_and_transitive_ancestors() -> None:
    """The DAG rejects cycles and exposes all predecessors to repair."""
    graph = PriorityDAG(('D1', 'D2', 'D3'))
    first = graph.add('D1', 'D2')
    assert first is not None
    second = first.add('D2', 'D3')
    assert second is not None
    assert second.ancestors()['D3'] == {'D1', 'D2'}
    assert second.add('D3', 'D1') is None
    assert second.topological() == ('D1', 'D2', 'D3')


def test_capacity_two_conflict_needs_an_incomparable_pair() -> None:
    """PBS orders a member of the violating capacity-plus-one set."""
    simulation = _network(3, 2)
    engine = SpaceTimeAStar(simulation.network)
    paths = {}
    for name in ('D1', 'D2', 'D3'):
        path = engine.find_path(name)
        assert path is not None
        paths[name] = path
    conflict = ConflictDetector(simulation.network).find(paths)[0]
    assert conflict.capacity == 2
    assert len(conflict.agents) == 3
    ancestors = PriorityDAG(('D1', 'D2', 'D3')).ancestors()
    assert PBSSolver._pair(conflict, ancestors) == ('D1', 'D2')


def test_pbs_resolves_capacity_two_and_executes() -> None:
    """The resulting schedule is legal through every simulation turn."""
    simulation = _network(4, 2)
    solver = simulation.solver
    assert isinstance(solver, PBSSolver)
    solver.prepare(simulation.network, simulation.drones)
    assert solver.solution is not None
    assert not ConflictDetector(simulation.network).find(solver.solution.paths)
    assert simulation.run().turns == 3


def test_zero_node_budget_returns_feasible_floor() -> None:
    """A search cutoff never loses the already valid schedule."""
    simulation = _network(4, 1)
    result = PBSSolver(node_budget=0).solve(
        simulation.network, [drone.id for drone in simulation.drones],
        budget=SearchBudget(),
    )
    assert result.status == 'node_limit'
    assert not ConflictDetector(simulation.network).find(result.paths)
