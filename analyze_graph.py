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


def dfs(graph: Graph, start: Node) -> tuple[Graph, dict[Node, int]]:
    """Perform depth-first search (DFS) on the given graph starting from the specified node.

    Return the DFS tree, which is a new graph representing the DFS tree, and a dictionary mapping nodes to their depths."""

    def dfs_recursive(
        node: Node,
        visited: set[Node],
        dfs_tree: Graph,
        depth: int,
        depths: dict[Node, int],
    ) -> None:
        visited.add(node)
        node_clone = node.clone(include_neighbors=False)
        node_clone.name = f"{node.name} [d={depth}]"
        # dfs_tree.add_node(node_clone)

        depths[node_clone] = depth
        for neighbor in sorted(node.neighbors):
            if neighbor not in visited:
                neighbor_clone = neighbor.clone(include_neighbors=False)
                neighbor_clone.name = f"{neighbor.name} [d={depth + 1}]"
                # if neighbor_clone not in dfs_tree.nodes:
                #     dfs_tree.add_node(neighbor_clone)
                dfs_tree.add_edge(
                    node_clone,
                    neighbor_clone,
                    symmetric=False,
                )
                dfs_recursive(neighbor, visited, dfs_tree, depth + 1, depths)

    visited = set()
    dfs_tree = Graph()
    depths = {}
    dfs_recursive(start, visited, dfs_tree, 0, depths)
    return dfs_tree, depths


def bfs(graph: Graph, start: Node) -> Graph:
    """
    Perform a breadth-first search (BFS) on the given graph starting from the specified node.
    Args:
        graph (Graph): The graph to perform BFS on.
        start (Node): The starting node for the BFS.
    Returns:
        Graph: A new graph representing the BFS tree.
    The BFS tree contains nodes and edges discovered during the BFS traversal. Each node in the BFS tree
    is a clone of the corresponding node in the original graph, but without neighbors initially. Edges
    are added to the BFS tree as they are discovered during the traversal.
    """

    visited = set()
    stack = [start]
    bfs_tree = Graph()

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            node_clone = node.clone(include_neighbors=False)
            if node_clone not in bfs_tree.nodes:
                bfs_tree.add_node(node_clone)
            for neighbor in sorted(node.neighbors):
                if neighbor not in visited:
                    neighbor_clone = neighbor.clone(include_neighbors=False)
                    if neighbor_clone not in bfs_tree.nodes:
                        bfs_tree.add_node(neighbor_clone)
                    stack.append(neighbor)
                    bfs_tree.add_edge(node_clone, neighbor_clone)

    return bfs_tree


def find_critical_nodes(parsed_snapshot: ParsedSnapshot) -> Counter[str]:
    counter: Counter[str] = Counter()
    raise NotImplementedError("TODO: implement find_critical_nodes")


if __name__ == "__main__":
    from pathlib import Path
    from rich.pretty import pprint

    parsed_snapshot = ParsedSnapshot(Path(__file__).parent / "logs")
    graph = parsed_snapshot.graph(positions=visualize.positions)
    graph.plot(
        show=True,
        detail=False,
    )

    dfs_tree, depths = dfs(graph, graph.node_list()[0])
    for k, v in depths.items():
        print(f"{k.name=} {v=}")

    # for index, node in enumerate(dfs_tree.node_list()):
    #     print(f"node[{index}]: {node} {len(node.neighbors)=}")
    #     for neighbor_index, neighbor in enumerate(node.neighbors):
    #         print(f"  neighbors[{neighbor_index}] {neighbor}")
    # print("edges:")
    # for index, (start, end) in enumerate(dfs_tree.edges):
    #     print(f"{index=} {start.name!r}->{end.name!r}")
    print(f"{dfs_tree=}")
    # for index, node in enumerate(dfs_tree):
    #     print(f"{index=} {node.name=}")

    for index, edge in enumerate(dfs_tree.edges):
        print(f"{index=} {edge[0].name!r} -> {edge[1].name!r}")
    dfs_tree.plot(
        show=True,
    )
    # waypoint_nodes = find_waypoint_nodes(parsed_snapshot)
    # print(f"waypoint_nodes=")
    # pprint(waypoint_nodes)
