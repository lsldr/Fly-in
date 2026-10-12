# Unsolvable-map checks

Every `.txt` file here is syntactically valid. Files `01` through `05` must
report an unreachable destination during planning or simulation, exit nonzero,
and terminate without a traceback or an infinite loop. They must pass
`--validate`: reachability is a planning concern, not a parsing error.

| Map | Case | Expected simulation result |
| --- | --- | --- |
| `01_no_connections.txt` | Start and goal with no edges | Unreachable |
| `02_isolated_start.txt` | Goal has a neighbor; start is isolated | Unreachable |
| `03_isolated_goal.txt` | Reachable chain and dead end; goal is isolated | Unreachable |
| `04_separate_cycles.txt` | Start and goal in separate cyclic components | Unreachable; cycle traversal terminates |
| `05_blocked_bridge.txt` | Only connecting route enters a blocked hub | Unreachable through accessible hubs |
| `06_reachable_control.txt` | Start reaches goal; spare hub is isolated | Success in 3 turns |

From the repository root, validate and then run a case:

```bash
uv run fly-in maps/unsolvable/04_separate_cycles.txt --validate
uv run fly-in maps/unsolvable/04_separate_cycles.txt --display moves --no-color
```

The first command should succeed. The second should report that no route
exists from `start` to `goal` and exit nonzero. A timeout can guard manual
testing against a regression that hangs:

```bash
timeout 10s uv run fly-in maps/unsolvable/04_separate_cycles.txt --display moves --no-color
echo $?
```

Expected exit status for the current CLI is `2`; status `124` means the
timeout expired. Repeat with `--solver greedy`, `--solver priority`, or other
solver choices to compare error handling. Use `06_reachable_control.txt` to
check that the program only requires a route between the endpoints, rather
than requiring every hub to belong to one connected component.
