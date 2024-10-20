from collections import Counter
import itertools
from parsed_snapshot import ParsedSnapshot


def find_all_paths(
    parsed_snapshot: ParsedSnapshot,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Find all the active instamesh routing paths, considering all possible `(source, destination)` pairs

    Return value is a mapping of `(source, destination)` pairs to tuples of hop nodes, including source and destination

    Given this network:
    ```
    +---+         +---+         +---+
    | A |  <--->  | B |  <--->  | C |
    +---+         +---+         +---+
    ```
    return value would be

    ```
    {
        ("A", "B"): ("A", "B"),
        ("A", "C"): ("A", "B", "C"),
        ("B", "A"): ("B", "A"),
        ("B", "C"): ("B", "C"),
        ("C", "A"): ("C", "B", "A"),
        ("C", "B"): ("C", "B"),
    }
    ```

    Args:
        parsed_snapshot (ParsedSnapshot): The snapshot

    Returns:
    dict[tuple[str, str], tuple[str, ...]]: A dict where keys are `(source, destination)` tuples
    and values are tuples of the path, including the endpoints
    """
    all_nodes = sorted(parsed_snapshot.parsed_logs)
    combos = list(itertools.product(all_nodes, repeat=2))
    paths = {}
    for index, (source, destination) in enumerate(combos):
        path = parsed_snapshot.trace_route(
            source=source,
            destination=destination,
        )
        path_nodes = [source] + [x.next_hop_node for x in path]
        paths[(source, destination)] = tuple(path_nodes)
    return paths


def find_waypoint_nodes(parsed_snapshot: ParsedSnapshot) -> Counter[str]:
    counter: Counter[str] = Counter()
    for serial in parsed_snapshot.parsed_logs:
        counter[serial] = 0

    for (src, dest), path_nodes in find_all_paths(parsed_snapshot).items():
        if len(path_nodes) > 2:
            waypoint_nodes = path_nodes[1:-1]
        else:
            waypoint_nodes = []
        for waypoint_node in waypoint_nodes:
            counter[waypoint_node] += 1

        print(f"{path_nodes=} {waypoint_nodes=}")
    return counter

def find_critical_nodes(parsed_snapshot: ParsedSnapshot) -> Counter[str]:
    counter: Counter[str] = Counter()
    raise NotImplementedError("TODO: implement find_critical_nodes")


if __name__ == "__main__":
    from pathlib import Path
    from rich.pretty import pprint

    parsed_snapshot = ParsedSnapshot(Path(__file__).parent / "logs")
    waypoint_nodes = find_waypoint_nodes(parsed_snapshot)
    print(f"waypoint_nodes=")
    pprint(waypoint_nodes)
