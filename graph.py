from collections import Counter
import dataclasses
import math
import functools
import random
from typing import ClassVar
from collections.abc import Iterable

import matplotlib.pyplot as plt


@functools.total_ordering
@dataclasses.dataclass(frozen=True, unsafe_hash=True)
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
    """

    x: float
    y: float

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


@dataclasses.dataclass(kw_only=True)
class Node:
    id: int = dataclasses.field(init=False, repr=True)
    point: Point
    name: str = dataclasses.field(default=None)
    infrastructure: bool = False
    neighbors: dict["Node", float] = dataclasses.field(
        default_factory=dict, compare=False, hash=False, repr=False
    )

    __next_id: ClassVar[int] = 0

    def __post_init__(self):
        self.id = Node.__next_id
        Node.__next_id += 1

        if self.name is None:
            self.name = f"node-{self.id:03}"

    def __hash__(self):
        return hash(self.id)

    def add_neighbor(
        self, *, node: "Node", weight: float, symmetric: bool = False
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
        if node in self.neighbors:
            raise ValueError(f"Node {node.name} is already a neighbor.")
        self.neighbors[node] = weight
        if symmetric:
            node.neighbors[self] = weight


class Graph:
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

        max_distance = math.sqrt(
            desired_neighbors / ((num_nodes - 1) * math.pi)
        )
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
        for node in nodes:
            for other in nodes:
                if (
                    node is not other
                    and (node not in other.neighbors)
                    and node.point.distance_to(other.point) < max_distance
                ):
                    node.add_neighbor(
                        node=other, weight=node.point.distance_to(other.point)
                    )
        return cls(nodes=nodes)

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

        row_spacing = 1.0 / (rows + 1)
        col_spacing = 1.0 / (cols + 1)

        max_distance = max(row_spacing, col_spacing) * max_distance_multiplier

        points = [
            Point(
                x=(col + 1) * col_spacing
                + rand.gauss(0, position_error_sigma) * col_spacing,
                y=(row + 1) * row_spacing
                + rand.gauss(0, position_error_sigma) * row_spacing,
            )
            for row in range(rows)
            for col in range(cols)
        ]

        nodes = tuple(
            Node(
                point=point,
                infrastructure=True,
                name=f"infra-{index:03}",
            )
            for index, point in enumerate(points)
        )
        for node in nodes:
            for other in nodes:
                if (
                    node is not other
                    and (node not in other.neighbors)
                    and node.point.distance_to(other.point) < max_distance
                ):
                    node.add_neighbor(
                        node=other, weight=node.point.distance_to(other.point)
                    )
        return cls(nodes=nodes)

    def __init__(self, nodes: Iterable[Node] = ()):
        self.nodes = set(nodes)

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
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Create a scatterplot of the nodes using matplotlib.

        Args:
            ax (plt.Axes | None, optional): A matplotlib Axes instance to plot on. Defaults to None.
            fig (plt.Figure | None, optional): A matplotlib Figure instance to plot on. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.

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

        # Plot lines for neighbors
        edge_counter = Counter()
        node_counter = Counter()
        for node in self.nodes:
            node_kind = "infra" if node.infrastructure else "mobile"
            node_counter[node_kind] += 1
            if node_kind == "infra":
                node_color = "#000000"
            else:
                node_color = "#ff0000"
            ax.scatter(node.point.x, node.point.y, color=node_color)

            for neighbor, weight in node.neighbors.items():
                if node.infrastructure and neighbor.infrastructure:
                    edge_kind = "infra-infra"
                    edge_color = "#000000"
                elif not node.infrastructure and not neighbor.infrastructure:
                    edge_kind = "mobile-mobile"
                    edge_color = "#ff0000"
                else:
                    edge_kind = "mobile-infra"
                    edge_color = "#00ff00"
                edge_counter[edge_kind] += 1
                ax.plot(
                    [node.point.x, neighbor.point.x],
                    [node.point.y, neighbor.point.y],
                    color=edge_color,
                    alpha=0.1,
                )

        for i, label in enumerate(labels):
            ax.annotate(label, (xs[i], ys[i]))

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(
            f"nodes={len(self.nodes):,}, edges={sum(edge_counter.values()):,}\n"
            + "Nodes: "
            + ", ".join(f"{key!r}={value:,}" for key, value in node_counter.items())
            + "\nEdges: "
            + ", ".join(f"{key!r}={value:,}" for key, value in edge_counter.items())
        )

        if show:
            plt.show()

        return fig, ax


if __name__ == "__main__":
    rand = random.Random(40351)
    points = [Point.random_point(rand=rand) for _ in range(10)]
    from rich.pretty import pprint

    pprint(sorted(points))

    # xs = [point.x for point in points]
    # ys = [point.y for point in points]
    # label = [f"points[{index}] {points[index].x:.2f}, {points[index].y:.2f}" for index in range(len(points))]

    # import matplotlib.pyplot as plt
    # plt.scatter(xs, ys)
    # plt.legend()
    # for index, label in enumerate(label):
    #     plt.annotate(label, (xs[index], ys[index]))
    # plt.show()

    node1 = Node(point=Point(0, 0))

    node2 = Node(point=Point(1, 1))

    node1.add_neighbor(node=node2, weight=1, symmetric=True)

    print(f"{node1=}")
    print(f"{node2=}")

    infra_graph = Graph.random_infrastructure(25, rand=rand)
    mobile_graph = Graph.random_mobile(
        100,
        rand=rand,
        # desired_neighbors=10,
    )
    graph = infra_graph
    # graph = mobile_graph
    for index, node in enumerate(graph):
        print(f"{index=}, ", end="")
        pprint(node)

    graph.plot()
