from collections import Counter, defaultdict
import dataclasses
import itertools
import math
import functools
import random
from typing import ClassVar, Literal
from collections.abc import Iterable

import matplotlib.pyplot as plt


@functools.total_ordering
@dataclasses.dataclass(frozen=False, unsafe_hash=True)
class Point:
    """
    A class to represent a point in 2D space.
    Attributes:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
    Methods:
        random_point(cls, x_range: tuple[float, float] = (0, 1), y_range: tuple[float, float] = (0, 1)) -> 'Point':
            Generates a random point within the given x and y ranges.
        distance_to(self, other: 'Point') -> float:
            Calculates the Euclidean distance between this point and another point.
        __lt__(self, other: 'Point') -> bool:
            Compares this point with another point for ordering based on y-coordinate, then x-coordinate.
        __eq__(self, other: 'Point') -> bool:
            Checks if this point is equal to another point based on x and y coordinates.
        __mul__(self, scalar: float) -> 'Point':
            Multiplies the coordinates of this point by a scalar.
        scale(self, origin: 'Point', scale: float) -> 'Point':
            Creates a new point, scaled from the given origin by the given scale.
    """

    x: float
    y: float

    def __post_init__(self):
        if self.x < 0:
            self.x = 0
        elif self.x > 1:
            self.x = 1
        if self.y < 0:
            self.y = 0
        elif self.y > 1:
            self.y = 1

    @classmethod
    def random_point(
        cls,
        *,
        x_range: tuple[float, float] = (0, 1),
        y_range: tuple[float, float] = (0, 1),
        rand: random.Random | None = None,
    ) -> "Point":
        """
        Generate a random point within the specified ranges.

        Args:
            x_range (tuple[float, float], optional): The range for the x-coordinate. Defaults to (0, 1).
            y_range (tuple[float, float], optional): The range for the y-coordinate. Defaults to (0, 1).
            rand (random.Random | None, optional): A random number generator instance. Defaults to None.

        Returns:
            Point: A Point object with random x and y coordinates within the specified ranges.
        """
        rand = rand or random
        x = rand.uniform(*x_range)
        y = rand.uniform(*y_range)
        return Point(x, y)

    def distance_to(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __lt__(self, other: "Point") -> bool:
        return (self.y, self.x) < (other.y, other.x)

    def __eq__(self, other: "Point") -> bool:
        return (self.x, self.y) == (other.x, other.y)

    def __mul__(self, scalar: float) -> "Point":
        return Point(self.x * scalar, self.y * scalar)

    def scale(self, origin: "Point", scale: float) -> "Point":
        """
        Create a new point, scaled from the given origin by the given scale.

        Args:
            origin (Point): The origin point to scale from.
            scale (float): The scale factor.

        Returns:
            Point: A new Point object that is scaled from the origin.
        """
        return Point(
            x=origin.x + (self.x - origin.x) * scale,
            y=origin.y + (self.y - origin.y) * scale,
        )


@functools.total_ordering
@dataclasses.dataclass(kw_only=True)
class Node:
    id: int | None = dataclasses.field(default=None, repr=True)
    point: Point
    name: str = dataclasses.field(default=None)
    infrastructure: bool = False
    neighbors: dict["Node", float] = dataclasses.field(
        default_factory=dict, compare=False, hash=False, repr=False
    )

    __next_id: ClassVar[int] = 0

    def clone(
        self,
        include_neighbors: bool = True,
    ) -> "Node":
        new_node = Node(
            id=self.id,
            point=self.point,
            name=self.name,
            infrastructure=self.infrastructure,
        )
        if include_neighbors:
            new_node.neighbors = self.neighbors[:]
        return new_node

    def __post_init__(self):
        if self.id is None:
            self.id = Node.__next_id
            Node.__next_id += 1

        if self.name is None:
            self.name = f"node-{self.id:03}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "Node") -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __lt__(self, other: "Node") -> bool:
        return self.id < other.id

    def add_neighbor(
        self,
        *,
        node: "Node",
        weight: float = 1,
        symmetric: bool = False,
    ) -> None:
        """
        Add a neighbor to this node with the given weight.

        Args:
            node (Node): The neighboring node to add.
            weight (float): The weight of the edge to the neighboring node.
            symmetric (bool, optional): If True, also add this node as a neighbor to the given node with the same weight. Defaults to False.

        Raises:
            ValueError: If the node is already a neighbor.
        """
        # if node in self.neighbors:
        #     return
        #     # raise ValueError(f"Node {node.name} is already a neighbor.")
        self.neighbors[node] = weight
        # print(f"[add_neighbor DEBUG] Added {node.name!r} as neighbor of {self.name!r}")

        if symmetric:
            node.add_neighbor(node=self, weight=weight, symmetric=False)
            # node.neighbors[self] = weight
            # print(
            #     f"[add_neighbor DEBUG] Added {self.name!r} as neighbor of {node.name!r} (sym)"
            # )

    def is_neighbor(self, node: "Node") -> bool:
        """
        Check if the given node is a neighbor of this node.

        Args:
            node (Node): The node to check.

        Returns:
            bool: True if the node is a neighbor, False otherwise.
        """
        return node in self.neighbors


default_colors = defaultdict(
    lambda: "#666666",
    **{
        # "infrastructure": "#3cb44b",
        "infrastructure": "#911eb4",
        "mobile": "#42d4f4",
        "mixed": "#9A6324",
    },
)


class Graph:
    def get(self, name: str) -> Node | None:
        for node in self.nodes:
            if node.name == name:
                return node

    def merge(
        self,
        other: "Graph",
        *,
        max_infrastructure_distance: float | None = None,
        max_mobile_distance: float | None = None,
        max_mixed_distance: float | None = None,
        ensure_close_nodes_are_neighbors: bool = False,
    ) -> "Graph":
        max_infrastructure_distance = (
            max_infrastructure_distance
            or self.max_infrastructure_distance
            or other.max_infrastructure_distance
        )
        max_mobile_distance = (
            max_mobile_distance or self.max_mobile_distance or other.max_mobile_distance
        )
        max_mixed_distance = (
            max_mixed_distance or self.max_mixed_distance or other.max_mixed_distance
        )
        if (
            max_mixed_distance is None
            and max_infrastructure_distance is not None
            and max_mobile_distance is not None
        ):
            max_mixed_distance = (max_infrastructure_distance + max_mobile_distance) / 2

        self_clone = self.clone()
        other_clone = other.clone()

        rv = Graph(
            nodes=self_clone.nodes.union(other_clone.nodes),
            max_infrastructure_distance=max_infrastructure_distance,
            max_mobile_distance=max_mobile_distance,
            max_mixed_distance=max_mixed_distance,
        )
        if ensure_close_nodes_are_neighbors:
            rv.ensure_close_nodes_are_neighbors()

        return rv

    @classmethod
    def random_mobile(
        cls,
        num_nodes: int,
        *,
        rand: random.Random | None = None,
        desired_neighbors: float = 10,
    ) -> "Graph":
        """
        Generate a random mobile node graph with the specified number of nodes.

        Args:
            num_nodes (int): The number of nodes in the graph.
            rand (random.Random | None, optional): A random number generator instance. Defaults to None.

        Returns:
            Graph: A Graph object with randomly generated nodes and edges.
        """
        rand = rand or random

        max_distance = math.sqrt(desired_neighbors / ((num_nodes - 1) * math.pi))
        # print(f"{max_distance=}")
        # exit()

        points = [Point.random_point(rand=rand) for _ in range(num_nodes)]

        nodes = tuple(
            Node(
                point=point,
                infrastructure=False,
                name=f"mobile-{index:03}",
            )
            for index, point in enumerate(points)
        )
        # for node in nodes:
        #     for other in nodes:
        #         if (
        #             node is not other
        #             and (node not in other.neighbors)
        #             and node.point.distance_to(other.point) < max_distance
        #         ):
        #             node.add_neighbor(
        #                 node=other, weight=node.point.distance_to(other.point)
        #             )
        rv = cls(nodes=nodes, max_mobile_distance=max_distance)
        rv.ensure_close_nodes_are_neighbors()
        return rv

    @classmethod
    def random_infrastructure(
        cls,
        num_nodes: int,
        *,
        rand: random.Random | None = None,
        max_distance_multiplier: float = 2.5,
        position_error_sigma: float = 0.15,
    ) -> "Graph":
        """
        Generate a random mobile node graph with the specified number of nodes.

        Args:
            num_nodes (int): The number of nodes in the graph.
            rand (random.Random | None, optional): A random number generator instance. Defaults to None.

        Returns:
            Graph: A Graph object with randomly generated nodes and edges.
        """
        rand = rand or random

        rows = math.floor(math.sqrt(num_nodes))
        cols = math.ceil(num_nodes / rows)

        row_spacing = 1.0 / (rows)
        col_spacing = 1.0 / (cols)

        max_distance = max(row_spacing, col_spacing) * max_distance_multiplier

        points = []
        for index in range(num_nodes):
            row = index // cols
            col = index % cols
            points.append(
                Point(
                    x=(col + 0.5) * col_spacing
                    + rand.gauss(0, position_error_sigma) * col_spacing,
                    y=(row + 0.5) * row_spacing
                    + rand.gauss(0, position_error_sigma) * row_spacing,
                )
            )

        nodes = tuple(
            Node(
                point=point,
                infrastructure=True,
                name=f"infra-{index:03}",
            )
            for index, point in enumerate(points)
        )
        # for node in nodes:
        #     for other in nodes:
        #         if (
        #             node is not other
        #             and (node not in other.neighbors)
        #             and node.point.distance_to(other.point) < max_distance
        #         ):
        #             node.add_neighbor(
        #                 node=other,
        #                 weight=node.point.distance_to(other.point),
        #                 symmetric=True,
        #             )

        rv = cls(
            nodes=nodes,
            max_infrastructure_distance=max_distance,
        )
        rv.ensure_close_nodes_are_neighbors()
        return rv

    def __init__(
        self,
        *,
        nodes: Iterable[Node] = (),
        max_infrastructure_distance: float | None = None,
        max_mobile_distance: float | None = None,
        max_mixed_distance: float | None = None,
    ):
        self.nodes = set(nodes)
        self.max_infrastructure_distance = max_infrastructure_distance
        self.max_mobile_distance = max_mobile_distance
        self.max_mixed_distance = max_mixed_distance

    @property
    def edges(self) -> list[tuple[Node, Node]]:
        def impl():
            for node in self.node_list():
                for neighbor in node.neighbors:
                    yield (node, neighbor)

        return list(impl())

    def node_list(self) -> list[Node]:
        return sorted(self.nodes)

    def ensure_close_nodes_are_neighbors(self, weight: float = 1) -> None:
        """Ensure that nodes that are close together are neighbors with the given weight."""
        for node1, node2 in itertools.combinations(self.nodes, 2):
            distance = node1.point.distance_to(node2.point)
            if node1.infrastructure and node2.infrastructure:
                max_distance = self.max_infrastructure_distance
            elif not node1.infrastructure and not node2.infrastructure:
                max_distance = self.max_mobile_distance
            else:
                max_distance = self.max_mixed_distance

            if max_distance is not None and distance < max_distance:
                if node1.is_neighbor(node2):
                    continue
                node1.add_neighbor(
                    node=node2,
                    weight=weight,
                    symmetric=True,
                )

    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph.

        Args:
            node (Node): The node to add.
        """
        self.nodes.add(node)

    def remove_node(self, node: Node) -> None:
        """
        Remove a node from the graph.

        Args:
            node (Node): The node to remove.
        """
        self.nodes.discard(node)

    def add_edge(
        self,
        start: Node,
        end: Node,
        weight: float = 1,
        symmetric: bool = True,
    ) -> None:
        """
        Add an edge between two nodes with the given weight.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.
            weight (float, optional): The weight of the edge. Defaults to 1.
            symmetric (bool, optional): If True, add the edge in both directions. Defaults to True.
        """
        if start not in self.nodes:
            self.add_node(start)
        if end not in self.nodes:
            self.add_node(end)
        start.add_neighbor(
            node=end,
            weight=weight,
            symmetric=symmetric,
        )

    def clone(self) -> "Graph":
        """
        Create a deep copy of the graph.

        Returns:
            Graph: A new Graph object that is a deep copy of the current graph.
        """
        new_nodes = {
            node: Node(
                point=node.point, name=node.name, infrastructure=node.infrastructure
            )
            for node in self.nodes
        }
        for node in self.nodes:
            for neighbor, weight in node.neighbors.items():
                new_nodes[node].add_neighbor(
                    node=new_nodes[neighbor], weight=weight, symmetric=False
                )
        return Graph(
            nodes=new_nodes.values(),
            max_infrastructure_distance=self.max_infrastructure_distance,
            max_mobile_distance=self.max_mobile_distance,
            max_mixed_distance=self.max_mixed_distance,
        )

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"Graph(nodes={list(self.nodes)})"

    def plot(
        self,
        *,
        ax: plt.Axes | None = None,
        fig: plt.Figure | None = None,
        show: bool = True,
        plot_edges: bool = True,
        show_annotations: bool = True,
        override_color: str | None = None,
        include_nodes: tuple[Literal["infrastructure", "mobile", "mixed"]] = (
            "infrastructure",
            "mobile",
            "mixed",
            "node",
        ),
        include_edges: tuple[Literal["infrastructure", "mobile", "mixed"]] = (
            "infrastructure",
            "mobile",
            "mixed",
            "edge",
        ),
        detail: bool = True,
        highlight_nodes: Iterable[Node] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Create a scatterplot of the nodes using matplotlib.

        Args:
            ax (plt.Axes | None, optional): A matplotlib Axes instance to plot on. Defaults to None.
            fig (plt.Figure | None, optional): A matplotlib Figure instance to plot on. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.
            plot_edges (bool, optional): Whether to plot edges between nodes. Defaults to True.
            show_annotations (bool, optional): Whether to show annotations for the nodes. Defaults to True.

        Returns:
            tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes objects.

        Raises:
            ValueError: If only one of fig or ax is provided.
        """
        if (ax is None) != (fig is None):
            raise ValueError(
                "Both 'fig' and 'ax' must be provided together, or neither."
            )

        xs = [node.point.x for node in self.nodes]
        ys = [node.point.y for node in self.nodes]
        labels = [node.name for node in self.nodes]

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)

        # Plot lines for neighbors if plot_edges is True
        edge_counter = Counter()
        node_counter = Counter()
        for node in self.nodes:
            node_kind = "node"

            if node.infrastructure:
                node_kind = "infrastructure"
            elif node.infrastructure is False:
                node_kind = "mobile"

            node_counter[node_kind] += 1
            node_color = override_color or default_colors[node_kind]
            if node_kind in include_nodes:
                ax.scatter(
                    node.point.x,
                    node.point.y,
                    color=node_color,
                    s=25,
                )

            if plot_edges:
                for neighbor, weight in node.neighbors.items():
                    if node.infrastructure is None or neighbor.infrastructure is None:
                        edge_kind = "edge"
                    elif node.infrastructure and neighbor.infrastructure:
                        edge_kind = "infrastructure"
                    elif not node.infrastructure and not neighbor.infrastructure:
                        edge_kind = "mobile"
                    else:
                        edge_kind = "mixed"
                    edge_color = override_color or default_colors[edge_kind]
                    edge_counter[edge_kind] += 1
                    if edge_kind in include_edges:
                        ax.plot(
                            [node.point.x, neighbor.point.x],
                            [node.point.y, neighbor.point.y],
                            color=edge_color,
                            # alpha=0.15,
                            alpha=0.2,
                            linewidth=1.5,
                            zorder=-1,
                        )

        if show_annotations:
            for i, label in enumerate(labels):
                ax.annotate(label, (xs[i], ys[i]))

        # ax.set_xlabel("X Coordinate")
        # ax.set_ylabel("Y Coordinate")
        if detail:
            ax.set_title(
                f"nodes={len(self.nodes):,}, edges={sum(edge_counter.values()):,}\n"
                + "Nodes: "
                + ", ".join(f"{key!r}={value:,}" for key, value in node_counter.items())
                + "\nEdges: "
                + ", ".join(f"{key!r}={value:,}" for key, value in edge_counter.items())
            )
        else:
            ax.set_title(
                f"nodes={len(self.nodes):,}, edges={sum(edge_counter.values()):,}"
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        

        if highlight_nodes:
            # xs = 
            for highlight_node in highlight_nodes:
                # print(f"DEBUG: Highlighting {highlight_node.point}")
                # ax.scatter(xs, ys, color="#ff000040", s=500, zorder=10)
                ax.scatter(
                    [highlight_node.point.x],
                    [highlight_node.point.y],
                    s=500,
                    color="#ff000040",
                    zorder=10,
                )


        if show:
            plt.show()
        return fig, ax


_rand = random.Random(40351)


realistic_dense_mesh = Graph.random_infrastructure(
    num_nodes=25,
    rand=_rand,
    # position_error_sigma=0.15,
    position_error_sigma=0.05,
    max_distance_multiplier=2.1,
).merge(
    Graph.random_mobile(
        num_nodes=125,
        rand=_rand,
        desired_neighbors=3,
    ),
    ensure_close_nodes_are_neighbors=True,
)


realistic_medium_mesh = Graph.random_infrastructure(
    num_nodes=16,
    rand=_rand,
    position_error_sigma=0.15,
    # position_error_sigma=0.05,
    max_distance_multiplier=1.9,
).merge(
    Graph.random_mobile(
        num_nodes=50,
        rand=_rand,
        desired_neighbors=1.5,
    ),
    ensure_close_nodes_are_neighbors=True,
)

realistic_sparse_mesh = Graph.random_infrastructure(
    num_nodes=9,
    rand=_rand,
    position_error_sigma=0.30,
    # position_error_sigma=0.05,
    max_distance_multiplier=1.4,
).merge(
    Graph.random_mobile(
        num_nodes=30,
        rand=_rand,
        desired_neighbors=1.5,
    ),
    ensure_close_nodes_are_neighbors=True,
)


if __name__ == "__main__":
    points = [Point.random_point(rand=_rand) for _ in range(10)]
    from rich.pretty import pprint

    pprint(sorted(points))
    node1 = Node(point=Point(0, 0))

    node2 = Node(point=Point(1, 1))

    node1.add_neighbor(node=node2, weight=1, symmetric=True)

    print(f"{node1=}")
    print(f"{node2=}")

    monochrome_graph = realistic_dense_mesh.clone()
    for node in monochrome_graph:
        node.infrastructure = None

    graph = realistic_dense_mesh
    graph = realistic_medium_mesh
    graph = realistic_sparse_mesh
    for index, node in enumerate(graph):
        print(f"{index=}, ", end="")
        pprint(node)

    fig, [[ax_all, ax_mixed], [ax_infra, ax_mobile]] = plt.subplots(
        2, 2, layout="constrained"
    )

    ax_all: plt.Axes
    ax_mixed: plt.Axes
    ax_infra: plt.Axes
    ax_mobile: plt.Axes

    override_color = "#000000"
    graph.plot(
        # plot_edges=False,
        fig=fig,
        ax=ax_all,
        show_annotations=False,
        show=False,
        override_color=override_color,
        include_edges=(
            "mixed",
            "infrastructure",
            "mobile",
        ),
        include_nodes=(
            "infrastructure",
            "mobile",
        ),
    )
    graph.plot(
        # plot_edges=False,
        fig=fig,
        ax=ax_mixed,
        show_annotations=False,
        show=False,
        # override_color=override_color,
        include_edges=("mixed",),
        include_nodes=(
            "infrastructure",
            "mobile",
        ),
    )
    graph.plot(
        # plot_edges=False,
        fig=fig,
        ax=ax_infra,
        show_annotations=False,
        show=False,
        # override_color=override_color,
        include_edges=(
            # "mixed",
            "infrastructure",
            # "mobile",
        ),
        include_nodes=(
            "infrastructure",
            "mobile",
        ),
    )
    graph.plot(
        # plot_edges=False,
        fig=fig,
        ax=ax_mobile,
        show_annotations=False,
        show=False,
        # override_color=override_color,
        include_edges=(
            # "mixed",
            # "infrastructure",
            "mobile",
        ),
        include_nodes=(
            "infrastructure",
            "mobile",
        ),
    )
    ax_all.set_title("FULL MESH\n" + ax_all.get_title().splitlines()[0])
    ax_infra.set_title("INFRASTRUCTURE EDGES\n" + ax_infra.get_title())
    ax_mobile.set_title("MOBILE EDGES\n" + ax_mobile.get_title())
    ax_mixed.set_title("MIXED EDGES\n" + ax_mixed.get_title())
    # ax_monochrome.remove()
    plt.show()
