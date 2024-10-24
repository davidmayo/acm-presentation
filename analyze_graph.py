from collections import Counter, deque
import itertools
import math

from matplotlib import pyplot as plt

from parsed_snapshot import ParsedSnapshot
from graph import (
    Graph,
    Point,
    Node,
    realistic_dense_mesh,
    realistic_medium_mesh,
    realistic_sparse_mesh,
)
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


def dfs(graph: Graph, start: Node) -> Graph:
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
        node_clone_name = f"{node.name} [d={depth}]"
        node_clone = dfs_tree.get(node_clone_name)
        if node_clone:
            print(f"{node_clone.name=} already in dfs_tree")
        else:
            node_clone = node.clone(include_neighbors=False)
            node_clone.name = f"{node.name} [d={depth}]"

        depths[node_clone] = depth
        for neighbor in sorted(node.neighbors):
            if neighbor not in visited:
                neighbor_clone_name = f"{neighbor.name} [d={depth + 1}]"
                neighbor_clone = dfs_tree.get(neighbor_clone_name)
                if neighbor_clone:
                    print(f"{neighbor_clone.name=} already in dfs_tree")
                else:
                    neighbor_clone = neighbor.clone(include_neighbors=False)
                    neighbor_clone.name = f"{neighbor.name} [d={depth + 1}]"
                    print(f"Creating {neighbor_clone.name=}")

                # if neighbor_clone not in dfs_tree.nodes:
                #     dfs_tree.add_node(neighbor_clone)
                print(
                    f"ADDING EDGE: {node_clone.name!r} -> {neighbor_clone.name!r} [n={len(dfs_tree.nodes)}, e={len(dfs_tree.edges)}]"
                )
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
    return dfs_tree


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


def find_articulation_points(graph: Graph) -> set[Node]:
    """
    Find all articulation points in the given graph.

    An articulation point (or cut vertex) is a node which, when removed, increases the number of connected components.

    Args:
        graph (Graph): The graph to find articulation points in.

    Returns:
        set[Node]: A set of nodes that are articulation points.
    """

    def dfs_articulation(
        node: Node,
        parent: Node,
        visited: set[Node],
        discovery: dict[Node, int],
        low: dict[Node, int],
        time: int,
        articulation_points: set[Node],
    ):
        visited.add(node)
        discovery[node] = low[node] = time
        children = 0

        for neighbor in node.neighbors:
            if neighbor == parent:
                continue
            if neighbor not in visited:
                children += 1
                dfs_articulation(
                    neighbor,
                    node,
                    visited,
                    discovery,
                    low,
                    time + 1,
                    articulation_points,
                )
                low[node] = min(low[node], low[neighbor])

                if parent is None and children > 1:
                    articulation_points.add(node)
                if parent is not None and low[neighbor] >= discovery[node]:
                    articulation_points.add(node)
            else:
                low[node] = min(low[node], discovery[neighbor])

    visited = set()
    discovery = {}
    low = {}
    articulation_points = set()

    for node in graph.nodes:
        if node not in visited:
            dfs_articulation(
                node=node,
                parent=None,
                visited=visited,
                discovery=discovery,
                low=low,
                time=0,
                articulation_points=articulation_points,
            )

    return articulation_points


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

    dfs_tree = dfs(graph, graph.node_list()[0])
    for index, edge in enumerate(dfs_tree.edges):
        print(f"{index=} {edge[0].name!r} -> {edge[1].name!r}")

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
    print("NODES:")
    for index, node in enumerate(dfs_tree.node_list()):
        print(f"{index=} {node.name=}")
    # waypoint_nodes = find_waypoint_nodes(parsed_snapshot)
    # print(f"waypoint_nodes=")
    # pprint(waypoint_nodes)

    graph = realistic_dense_mesh
    # graph.plot(
    #     show=True,
    # )

    articulation_points = find_articulation_points(graph)
    print(f"points=")
    pprint(articulation_points)
    root = graph.node_list()[0]
    dfs_tree = dfs(graph, root)
    fig, ax = dfs_tree.plot(
        # show_annotations=False,
        detail=False,
        override_color="black",
        show=False,
    )
    ax.scatter(root.point.x, root.point.y, color="red", s=500, zorder=10)
    plt.show()

    def circle_points(center: Point, radius: float, num_points: int) -> list[Point]:
        points = []
        for i in range(num_points):
            angle = 2 * 3.14159 * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append(Point(x, y))
        return points

    graph = Graph()
    center_points = circle_points(Point(0.5, 0.5), 0.25, 4)
    center_nodes = [Node(point=point) for point in center_points]
    for node1, node2 in itertools.combinations(center_nodes, r=2):
        graph.add_edge(node1, node2)

    for center_point, center_node in zip(center_points, center_nodes):
        outer_points = circle_points(center_point.scale(Point(0.5, 0.5), 1.5), 0.075, 6)
        outer_nodes = []
        for outer_point in outer_points:
            outer_node = Node(point=outer_point)
            outer_nodes.append(outer_node)
            graph.add_edge(center_node, outer_node)
        for outer_node1, outer_node2 in itertools.combinations(outer_nodes, r=2):
            graph.add_edge(outer_node1, outer_node2)

    articulation_points = find_articulation_points(graph)
    print(f"ARTICULATION POINTS:")
    pprint(articulation_points)

    print(f"NODES:")
    for index, node in enumerate(graph.node_list()):
        print(f"{index=} {node.name=}")

    graph.plot(
        show_annotations=False,
        show=False,
    )
    plt.show()

    dfs_tree = dfs(graph, graph.node_list()[0])
    dfs_tree.plot(
        show=True,
    )

    graph = realistic_sparse_mesh
    dfs_tree = dfs(graph, graph.node_list()[0])
    articulation_points = find_articulation_points(graph)
    print(f"ARTICULATION POINTS:")
    pprint(articulation_points)
    fig, ax = graph.plot(
        show=False,
        # override_color="black",
        show_annotations=False,
    )
    xs = [node.point.x for node in articulation_points]
    ys = [node.point.y for node in articulation_points]
    ax.scatter(xs, ys, color="#ff000040", s=500, zorder=10)
    for node in articulation_points:
        ax.text(node.point.x, node.point.y, node.name, fontsize=12)
    plt.show()
    # dfs_tree.plot(
    #     show=True,
    #     # show_annotations=False,
    #     override_color="#ff00ff",
    # )
