from pathlib import Path
import random

from matplotlib import pyplot as plt
import networkx as nx

from graph import Graph, Node, Point
from parsed_log import InstameshRoutingTableEntry, ParsedLog


class ParsedSnapshot:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.log_paths = sorted(self.path.glob("*.log"))
        self.parsed_logs: dict[str, ParsedLog] = {}
        for log_path in self.log_paths:
            parsed_log = ParsedLog(log_path)
            serial = parsed_log.serial_number
            if not serial:
                continue
            self.parsed_logs[serial] = parsed_log

    def graph(
        self,
        positions: dict[str, tuple[float, float]] | None = None
    ) -> Graph:
        graph = Graph()
        for serial in self.parsed_logs:
            if positions:
                point = Point(*positions[serial])
            else:
                point = Point(0.5, 0.5)
            graph.add_node(Node(
                point = point,
                name=serial,
                infrastructure=True,
            ))
        for serial, parsed_log in self.parsed_logs.items():
            for neighbor_serial, instamesh_neighbor_entry in parsed_log.instamesh_neighbors.items():
                this_node = graph.get(serial)
                neighbor_node = graph.get(neighbor_serial)

                if neighbor_node in this_node.neighbors:
                    continue

                if instamesh_neighbor_entry.cost is None:
                    weight = 1
                else:
                    weight = instamesh_neighbor_entry.cost
                this_node.add_neighbor(
                    node=neighbor_node,
                    weight=weight,
                    symmetric=True,
                )
                pass
                # graph.add_edge(serial, neighbor_serial)
        return graph

    def networkx_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for serial, parsed_log in self.parsed_logs.items():
            for neighbor_serial in parsed_log.instamesh_neighbors:
                graph.add_edge(serial, neighbor_serial)
        return graph

    def trace_route(
        self, source: str, destination: str, *, debug: bool = False
    ) -> list[InstameshRoutingTableEntry]:
        hop_count = 0
        path = []
        while source != destination:
            source_parsed_log = self.parsed_logs[source]
            source_routing_table = source_parsed_log.instamesh_routing_table
            assert source_routing_table
            next_hop = source_routing_table[destination]
            if debug:
                print(f"{hop_count=} {next_hop=}")
            source = next_hop.next_hop_node
            path.append(next_hop)
            hop_count += 1
        return path


if __name__ == "__main__":
    from rich.pretty import pprint

    path = Path(__file__).parent / "logs"
    parsed_snapshot = ParsedSnapshot(path)
    print("list(parsed_snapshot.parsed_logs)=")
    pprint(list(parsed_snapshot.parsed_logs))

    print()
    print("{parsed_snapshot.trace_route('A', 'M', debug=False)=")
    pprint(parsed_snapshot.trace_route("A", "M", debug=False))

    graph = parsed_snapshot.networkx_graph()
    # graph = nx.random_geometric_graph(50, 0.125, seed=40351,)

    print(f"{graph=!s}")
    print(f"{graph=!r}")

    # nx.draw(graph)
    # plt.show()

    random.seed(40351)
    for edge_index, edge in enumerate(graph.edges):
        # print(edge_index, edge)
        node0 = graph.nodes[edge[0]]
        if "pos" not in node0:
            node0["pos"] = [random.random(), random.random()]

        node1 = graph.nodes[edge[1]]
        if "pos" not in node1:
            node1["pos"] = [random.random(), random.random()]
        # pos = nodes[0]["pos"]
        print(f"{edge_index=} {edge=} {node0=} {type(node0)=}")

    # nx.draw(graph)
    positions = {
        "A": (76, 42),
        "B": (107, 135),
        "C": (248, 138),
        "D": (317, 114),
        "E": (564, 111),
        "F": (837, 40),
        "G": (198, 232),
        "H": (320, 187),
        "I": (449, 195),
        "J": (733, 167),
        "K": (161, 340),
        "L": (340, 371),
        "M": (475, 361),
        "N": (583, 291),
        "O": (851, 310),
        "P": (341, 496),
        "Q": (547, 442),
        "R": (823, 528),
        "S": (868, 455),
    }
    positions = {k: (v[0] / 922, (573 - v[1]) / 573) for k, v in positions.items()}
    for n in graph.nodes:
        print(f"{n=} {graph.nodes[n] = }")

    nx.draw_networkx(graph, pos=positions)
    plt.show()
    print(graph.nodes["R"])

    graph = parsed_snapshot.graph(
        positions=positions,
    )
    print(f"{graph=}")
    graph.plot(show=True, detail=False,)