"""Review fixtures and independent analytic bounds for large stress maps.

Run ``python tests/optimality_cases.py`` to regenerate only maps/extra.
Every added detour joins a single articulation and cannot shorten delivery.
Counts include start and goal. All large maps have 100 drones, 200 hubs,
and 300 undirected connections; the layered map also has useful crosslinks.
"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRA = ROOT / "maps" / "extra"
CATALOG_TURNS = {
    "easy/01_linear_path.txt": 4,
    "easy/02_simple_fork.txt": 4,
    "easy/03_basic_capacity.txt": 4,
    "medium/01_dead_end_trap.txt": 8,
    "medium/02_circular_loop.txt": 10,
    "medium/03_priority_puzzle.txt": 6,
    "hard/01_maze_nightmare.txt": 13,
    "hard/02_capacity_hell.txt": 16,
    "hard/03_ultimate_challenge.txt": 26,
    "challenger/01_the_impossible_dream.txt": 43,
}


@dataclass(frozen=True)
class AnalyticCase:
    """Independent necessary throughput bound with a matching pipeline."""

    filename: str
    latencies: tuple[int, ...]
    capacities: tuple[int, ...]
    proof: str

    def deliveries_by(self, turn: int) -> int:
        """Count maximal deliveries permitted by the specified cut windows."""
        return sum(capacity * max(0, turn - latency + 1)
                   for latency, capacity in
                   zip(self.latencies, self.capacities))

    def minimum(self, fleet: int = 100) -> int:
        """Find the first horizon the necessary cut bound can accommodate."""
        turn = 0
        while self.deliveries_by(turn) < fleet:
            turn += 1
        return turn


LARGE_CASES = (
    AnalyticCase("10_cyclic_funnel.txt", (20,), (1,),
                 "All deliveries cross the unit final edge, first at 20. "
                 "The chain sustains one per turn; attached cycles add "
                 "no route around the cut."),
    AnalyticCase("11_restricted_funnel.txt", (30,), (1,),
                 "Ten restricted entries add ten turns to the chain. "
                 "The final edge permits one arrival per turn from 30; "
                 "restricted arrivals release links for the next entry."),
    AnalyticCase("12_unequal_parallel_routes.txt", (8, 11, 14, 17),
                 (1, 2, 3, 1),
                 "Separate route exit cuts permit c*(T-latency+1) "
                 "deliveries each. At 25 only 93 fit; at 26 exactly 100 "
                 "fit and independent pipelining attains the bound."),
    AnalyticCase("13_restricted_shared_cut.txt", (11,), (2,),
                 "Five upstream routes feed a restricted capacity-two "
                 "hub. First delivery is 11; its occupancy cut allows "
                 "two per turn. At 59 at most 98 arrive; at 60, 100."),
    AnalyticCase("14_braided_layers.txt", (11,), (15,),
                 "Ten layers require eleven forward steps; crosslinks "
                 "stay within a layer. Fifteen final links permit fifteen "
                 "deliveries per turn from 11. Fifteen straight pipelines "
                 "attain 17 turns; 16 permits only 90 deliveries."),
)

HARD_CASE = AnalyticCase(
    "15_mass_priority_crossing.txt", (3,), (100,),
    "Shortest delivery needs three steps. Fifty drones take s-c-b-g "
    "and fifty take s-a-d-g, attaining three turns. The preferred "
    "s-a-b-g route consumes resources from both independent routes. "
    "Unlike count-only stress, this forces a wide capacity CT split.",
)

SMALL_NAMES = (
    "01_priority_crossing.txt", "02_flow_relaxation_gap.txt",
    "03_restricted_merge_loop.txt", "04_capacity_swap_ring.txt",
    "05_restricted_goal_blocked_shortcut.txt",
)


class MapBuilder:
    """Declare deterministic stress graphs without graph-library helpers."""

    def __init__(self) -> None:
        """Start one 100-drone graph with unlimited endpoints."""
        self.hubs = ["start_hub: s 0 0", "end_hub: g 30 0"]
        self.links: list[str] = []
        self.pairs: set[frozenset[str]] = set()

    def hub(self, name: str, kind: str = "normal", cap: int = 1) -> None:
        """Append a uniquely named hub with explicit resource metadata."""
        self.hubs.append(f"hub: {name} {len(self.hubs)} 1 "
                         f"[zone={kind} max_drones={cap}]")

    def link(self, first: str, second: str, cap: int = 1) -> None:
        """Append one undirected link, rejecting duplicate fixture edges."""
        pair = frozenset((first, second))
        assert len(pair) == 2 and pair not in self.pairs
        self.pairs.add(pair)
        self.links.append(f"connection: {first}-{second} "
                          f"[max_link_capacity={cap}]")

    def route(
        self, names: list[str], capacity: int = 1, restricted: int = 0,
    ) -> None:
        """Build a pipeline, charging extra latency on its first hubs."""
        for index, name in enumerate(names[1:-1]):
            self.hub(name, "restricted" if index < restricted
                     else "priority" if index % 3 == 0 else "normal",
                     capacity)
        for first, second in zip(names, names[1:]):
            self.link(first, second, capacity)

    def pad(self, anchor: str) -> None:
        """Add accessible cycles, genuine leaves, and blocked decoys.

        All padding joins exactly one existing anchor, so removing its
        round trips leaves a delivery walk at least as short. Chords join
        only padding hubs. Some isolated leaves are blocked, the rest
        remain accessible dead ends. No disconnected count-only padding.
        """
        amount = 200 - len(self.hubs)
        for index in range(amount):
            kind = ("blocked" if index >= amount - 5 else
                    "restricted" if index % 11 == 0 else "normal")
            self.hub(f"detour{index}", kind)
            self.link(anchor, f"detour{index}")
        chords = 300 - len(self.links)
        assert 0 <= chords < amount - 5
        for index in range(chords):
            self.link(f"detour{index}", f"detour{index + 1}")
        assert len(self.hubs) == 200 and len(self.links) == 300

    def text(self, proof: str) -> str:
        """Serialize hubs before links with the independent proof comment."""
        return "\n".join([f"# {proof}", "nb_drones: 100",
                          *self.hubs, *self.links]) + "\n"


def large_map(case: AnalyticCase) -> str:
    """Construct a graph with the stated independent cut/latency bound."""
    builder = MapBuilder()
    if case == HARD_CASE:
        for name in ("a", "b", "c", "d"):
            builder.hub(name, "priority" if name in ("a", "b")
                        else "normal", 50)
        for first, second in (("s", "a"), ("s", "c"), ("a", "b"),
                              ("a", "d"), ("c", "b"), ("b", "g"),
                              ("d", "g")):
            builder.link(first, second, 50)
        builder.pad("a")
    elif "funnel" in case.filename:
        restricted = 10 if "restricted" in case.filename else 0
        builder.route(["s", *[f"h{i}" for i in range(1, 20)], "g"],
                      restricted=restricted)
        builder.pad("h10")
    elif "parallel" in case.filename:
        for branch, (length, restricted, capacity) in enumerate(
            zip((6, 8, 10, 12), (2, 3, 4, 5), case.capacities)
        ):
            builder.route(["s", *[f"p{branch}h{i}"
                                  for i in range(1, length)], "g"],
                          capacity, restricted)
        builder.pad("p0h3")
    elif "shared_cut" in case.filename:
        builder.hub("merge", cap=10)
        builder.hub("restricted", "restricted", 2)
        for branch in range(5):
            builder.route(["s", *[f"p{branch}h{i}" for i in range(1, 8)],
                           "merge"], capacity=2)
        builder.link("merge", "restricted", 2)
        builder.link("restricted", "g", 2)
        builder.pad("merge")
    elif "braided" in case.filename:
        for lane in range(15):
            builder.route(["s", *[f"l{level}p{lane}"
                                  for level in range(10)], "g"])
        for level in range(6):
            for lane in range(15 if level < 5 else 12):
                builder.link(f"l{level}p{lane}",
                             f"l{level}p{(lane + 1) % 15}")
        builder.pad("l5p0")
    else:
        raise ValueError(case.filename)
    return builder.text(case.proof)


def write_extra_maps() -> None:
    """Regenerate only the explicitly requested review map fixtures."""
    EXTRA.mkdir(exist_ok=True)
    for output, source in zip(SMALL_NAMES[:2],
                              ("prioritized_trap.txt",
                               "flow_relaxation_gap.txt")):
        (EXTRA / output).write_text(
            (ROOT / "tests" / "fixtures" / source).read_text(
                encoding="utf-8"), encoding="utf-8",
        )
    small = (
        """nb_drones: 3
start_hub: s 0 0
end_hub: g 4 0
hub: a 1 0 [zone=priority]
hub: r 1 1 [zone=restricted max_drones=2]
hub: merge 3 0 [zone=restricted]
hub: loop 2 1
connection: s-a
connection: s-r [max_link_capacity=2]
connection: a-merge
connection: r-merge
connection: a-loop
connection: loop-r
connection: merge-g
""",
        """nb_drones: 3
start_hub: s 0 0
end_hub: g 3 0
hub: a 1 0 [max_drones=2]
hub: b 1 1 [max_drones=2]
hub: c 2 0
connection: s-a [max_link_capacity=2]
connection: s-b [max_link_capacity=2]
connection: a-b [max_link_capacity=2]
connection: a-c
connection: b-c
connection: c-g [max_link_capacity=2]
""",
        """nb_drones: 5
start_hub: s 0 0 [max_drones=0]
end_hub: g 3 0 [zone=restricted max_drones=ignored]
hub: blocked 1 0 [zone=blocked]
hub: a 1 1 [max_drones=2]
hub: b 2 1 [max_drones=2]
connection: s-blocked [max_link_capacity=100]
connection: blocked-g [max_link_capacity=100]
connection: s-a [max_link_capacity=2]
connection: a-b [max_link_capacity=2]
connection: b-g [max_link_capacity=2]
""",
    )
    for name, text in zip(SMALL_NAMES[2:], small):
        (EXTRA / name).write_text(text, encoding="utf-8")
    for case in (*LARGE_CASES, HARD_CASE):
        (EXTRA / case.filename).write_text(large_map(case), encoding="utf-8")


if __name__ == "__main__":
    write_extra_maps()
