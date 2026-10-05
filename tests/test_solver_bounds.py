"""Verify latency and capacity lower bounds independently of solvers."""

import json
from pathlib import Path
from random import Random
import subprocess
import sys

import pytest

from joint_oracle import exact_makespan

from fly_in.parser import Parser
from fly_in.realm.network import DroneNetwork
from fly_in.solvers.bounds import InstanceProfile, NetworkProfiler


ROOT = Path(__file__).resolve().parents[1]


def _profile(text: str) -> InstanceProfile:
    """Parse a self-contained fixture and calculate its static bounds."""
    config = Parser().parse_text(text)
    return NetworkProfiler(DroneNetwork.from_config(config)).profile(
        config.nb_drones,
    )


def test_single_connection_counts_unlimited_endpoints() -> None:
    """Three direct slots transport seven drones in at least three turns."""
    profile = _profile("""
nb_drones: 7
start_hub: s 0 0
end_hub: g 1 0
connection: s-g [max_link_capacity=3]
""")
    assert profile.shortest_distance == 1
    assert profile.bottleneck_capacity == 3
    assert profile.lower_bound == 3
    assert profile.density == 0.5


def test_vertex_capacity_is_included_in_min_cut() -> None:
    """A narrow intermediate hub limits otherwise wide connections."""
    profile = _profile("""
nb_drones: 5
start_hub: s 0 0
end_hub: g 2 0
hub: a 1 0 [max_drones=2]
connection: s-a [max_link_capacity=10]
connection: a-g [max_link_capacity=10]
""")
    assert profile.shortest_distance == 2
    assert profile.bottleneck_capacity == 2
    assert profile.lower_bound == 4


def test_restricted_link_releases_on_arrival() -> None:
    """Two-turn entrance increases latency without halving throughput."""
    profile = _profile("""
nb_drones: 4
start_hub: s 0 0
end_hub: g 2 0
hub: a 1 0 [zone=restricted]
connection: s-a
connection: a-g
""")
    assert profile.shortest_distance == 3
    assert profile.bottleneck_capacity == 1
    assert profile.lower_bound == 6
    assert profile.restricted_ratio == pytest.approx(1 / 3)


def test_restricted_goal_adds_its_own_latency() -> None:
    """The final extra transit turn contributes to the resource suffix."""
    profile = _profile("""
nb_drones: 3
start_hub: s 0 0
end_hub: g 1 0 [zone=restricted]
connection: s-g
""")
    assert profile.shortest_distance == 2
    assert profile.lower_bound == 4


def test_parallel_routes_use_aggregate_flow() -> None:
    """Two independent unit pipelines deliver four drones by turn three."""
    profile = _profile("""
nb_drones: 4
start_hub: s 0 0
end_hub: g 2 0
hub: a 1 0
hub: b 1 1
connection: s-a
connection: a-g
connection: s-b
connection: b-g
""")
    assert profile.bottleneck_capacity == 2
    assert profile.lower_bound == 3


def test_long_parallel_route_does_not_offer_immediate_capacity() -> None:
    """A late second route cannot inflate early delivery throughput."""
    profile = _profile("""
nb_drones: 4
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0
hub: b 1 1 [zone=restricted]
hub: c 2 1 [zone=restricted]
connection: s-a
connection: a-g
connection: s-b
connection: b-c
connection: c-g
""")
    assert profile.shortest_distance == 2
    assert profile.bottleneck_capacity == 2
    assert profile.lower_bound == 5


def test_blocked_and_disconnected_hubs_do_not_add_flow() -> None:
    """Map statistics retain unusable hubs but routing excludes them."""
    profile = _profile("""
nb_drones: 3
start_hub: s 0 0
end_hub: g 1 0
hub: b 0 1 [zone=blocked]
hub: isolated 9 9
connection: s-g
connection: s-b [max_link_capacity=10]
connection: b-g [max_link_capacity=10]
""")
    assert profile.vertices == 4
    assert profile.edges == 3
    assert profile.bottleneck_capacity == 1
    assert profile.lower_bound == 3


def test_disconnected_goal_is_reported() -> None:
    """Unreachable instances have no finite completion-time bound."""
    with pytest.raises(ValueError, match="unreachable"):
        _profile("""
nb_drones: 1
start_hub: s 0 0
end_hub: g 1 0
""")


def test_invalid_fleet_is_reported() -> None:
    """The public helper rejects nonpositive fleets independently."""
    config = Parser().parse_text("""
nb_drones: 1
start_hub: s 0 0
end_hub: g 1 0
connection: s-g
""")
    with pytest.raises(ValueError, match="positive"):
        NetworkProfiler(DroneNetwork.from_config(config)).profile(0)


def test_challenger_lower_bound_is_43() -> None:
    """The final unit-capacity cut proves the existing challenger optimum."""
    config = Parser().parse(
        ROOT / "maps/challenger/01_the_impossible_dream.txt",
    )
    profile = NetworkProfiler(DroneNetwork.from_config(config)).profile(
        config.nb_drones,
    )
    assert profile.shortest_distance == 19
    assert profile.bottleneck_capacity == 1
    assert profile.lower_bound == 43


def test_analyzer_json_is_machine_readable() -> None:
    """The standalone script exposes every instance feature as JSON."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_bounds.py"),
         str(ROOT / "maps/easy/01_linear_path.txt"), "--json"],
        check=True, capture_output=True, text=True,
    )
    profile = json.loads(result.stdout)
    assert profile["lower_bound"] == 4
    assert profile["shortest_distance"] == 3


def test_analyzer_rejects_unreachable_map(tmp_path: Path) -> None:
    """Expected input errors use a concise message without a traceback."""
    map_path = tmp_path / "disconnected.txt"
    map_path.write_text(
        "nb_drones: 1\nstart_hub: s 0 0\nend_hub: g 1 0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_bounds.py"),
         str(map_path)], capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "unreachable" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("seed", range(24))
def test_bound_does_not_exceed_independent_joint_optimum(seed: int) -> None:
    """Cross-check small mixed-capacity maps against exhaustive joint BFS."""
    random = Random(seed)
    fleet = 1 + seed % 3
    lines = [f"nb_drones: {fleet}", "start_hub: s 0 0", "end_hub: g 3 0"]
    for index, name in enumerate(("a", "b"), start=1):
        zone = "restricted" if random.getrandbits(1) else "normal"
        capacity = random.randint(1, 2)
        lines.append(f"hub: {name} {index} 0 "
                     f"[zone={zone} max_drones={capacity}]")
    edges = [("s", "a"), ("a", "b"), ("b", "g")]
    for edge in (("s", "b"), ("s", "g"), ("a", "g")):
        if random.getrandbits(1):
            edges.append(edge)
    for source, target in edges:
        lines.append(f"connection: {source}-{target} "
                     f"[max_link_capacity={random.randint(1, 2)}]")
    config = Parser().parse_text("\n".join(lines))
    network = DroneNetwork.from_config(config)
    lower_bound = NetworkProfiler(network).profile(fleet).lower_bound
    assert lower_bound <= exact_makespan(network, fleet)
