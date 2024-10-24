from collections import Counter, deque
import itertools

from parsed_snapshot import ParsedSnapshot
from graph import Graph, Point, Node
import visualize


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

def dfs2(
    graph: Graph,
    node: Node | None = None,
) -> Graph:
    node = node or graph.node_list()[0]
    assert node is not None

    visited = [node]
    output = [node]
    stack = [node]

    tree = Graph()

    node_clone = node.clone(include_neighbors=False)
    tree.add_node(node_clone)

    while stack:
        node = stack[-1]
        if node not in visited:
            output.append(node)
            tree.add_node(node.clone(include_neighbors=False))
            visited.append(node)
        remove_from_stack = True
        for next_node in node.neighbors:
            if next_node not in visited:
                stack.append(next_node)
                remove_from_stack = False
                break
        if remove_from_stack:
            removed = stack.pop()
            tree.get(removed.name).add_neighbor(node=tree.get(node.name), weight=1, symmetric=True)
    return output
    
    return dfs_tree

def dfs(
    graph: Graph,
    *,
    node: Node | None = None,
    visited: set[Node] | None = None,
    dfs_tree: Graph | None = None,
) -> Graph:
    if visited is None:
        visited = set()
    if dfs_tree is None:
        print(f"----- BEGIN DFS -----")
        dfs_tree = Graph()
    if node is None:
        node = graph.node_list()[0]
    visited.add(node)
    node_clone = node.clone(include_neighbors=False)
    if node_clone not in dfs_tree.nodes:
        dfs_tree.add_node(node_clone)
    print(f"DEBUG: DFS ADDING NODE {node_clone.name!r} ({len(dfs_tree.nodes)=}) ({len(dfs_tree.edges)=}) {id(dfs_tree)=}")
    for neighbor in sorted(node.neighbors):
        if neighbor in visited:
            continue
        neighbor_clone = neighbor.clone(include_neighbors=False)
        if neighbor_clone not in dfs_tree.nodes:
            dfs_tree.add_node(neighbor_clone)
        print(f"DEBUG: DFS ADD EDGE {node_clone.name!r}->{neighbor_clone.name!r} to {id(dfs_tree)}")
        node_clone.add_neighbor(
            node=neighbor_clone,
            symmetric=False,
            weight=1,
        )
        dfs_tree = dfs(
            graph=graph,
            node=neighbor,
            visited=visited,
            dfs_tree=dfs_tree,
        )
    return dfs_tree



def find_critical_nodes(parsed_snapshot: ParsedSnapshot) -> Counter[str]:
    counter: Counter[str] = Counter()
    raise NotImplementedError("TODO: implement find_critical_nodes")


if __name__ == "__main__":
    from pathlib import Path
    from rich.pretty import pprint

    parsed_snapshot = ParsedSnapshot(Path(__file__).parent / "logs")
    graph = parsed_snapshot.graph(positions=visualize.positions)
    graph.plot(show=True, detail=False,)
    
    dfs_tree = dfs2(graph)
    # for index, node in enumerate(dfs_tree.node_list()):
    #     print(f"node[{index}]: {node} {len(node.neighbors)=}")
    #     for neighbor_index, neighbor in enumerate(node.neighbors):
    #         print(f"  neighbors[{neighbor_index}] {neighbor}")
    # print("edges:")
    # for index, (start, end) in enumerate(dfs_tree.edges):
    #     print(f"{index=} {start.name!r}->{end.name!r}")
    print(f"{dfs_tree=}")
    for index, node in enumerate(dfs_tree):
        print(f"{index=} {node.name=}")
    # dfs_tree.plot(show=True,)
    # waypoint_nodes = find_waypoint_nodes(parsed_snapshot)
    # print(f"waypoint_nodes=")
    # pprint(waypoint_nodes)
