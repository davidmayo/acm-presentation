from collections import Counter
from matplotlib import pyplot as plt
from rich.pretty import pprint

from graph import Graph, Node, Point
from analyze_graph import dfs, find_articulation_points


if __name__ == "__main__":
    node_c = Node(name="C", point=Point(0.5, 0.6), infrastructure=True)
    node_a = Node(name="A", point=Point(0.3, 0.9), infrastructure=True)
    node_b = Node(name="B", point=Point(0.7, 0.9), infrastructure=True)
    node_d = Node(name="D", point=Point(0.5, 0.4), infrastructure=True)
    node_e = Node(name="E", point=Point(0.3, 0.1), infrastructure=True)
    node_f = Node(name="F", point=Point(0.7, 0.1), infrastructure=True)
    nodes = [
        node_c,
        node_a,
        node_b,
        node_d,
        node_e,
        node_f,

    ]
    # nodes = [
    #     Node(name="node-3", point=Point(0.5, 0.6), infrastructure=True),
    #     Node(name="node-1", point=Point(0.3, 0.9), infrastructure=True),
    #     Node(name="node-2", point=Point(0.7, 0.9), infrastructure=True),
    #     Node(name="node-4", point=Point(0.5, 0.4), infrastructure=True),
    #     Node(name="node-5", point=Point(0.3, 0.1), infrastructure=True),
    #     Node(name="node-6", point=Point(0.7, 0.1), infrastructure=True),
    # ]

    graph = Graph(nodes=nodes)
    graph.add_edge(nodes[0], nodes[1], symmetric=True)
    graph.add_edge(nodes[1], nodes[2], symmetric=True)
    graph.add_edge(nodes[2], nodes[0], symmetric=True)

    graph.add_edge(nodes[0 + 3], nodes[1 + 3], symmetric=True)
    graph.add_edge(nodes[1 + 3], nodes[2 + 3], symmetric=True)
    graph.add_edge(nodes[2 + 3], nodes[0 + 3], symmetric=True)

    graph.add_edge(nodes[0], nodes[3])
    articulation_point_nodes = find_articulation_points(graph=graph)

    graph.plot(
        detail=False,
        override_color="#000000",
        # highlight_nodes=articulation_point_nodes,
    )

    pprint(articulation_point_nodes, expand_all=True)

    dfs_tree = dfs(graph=graph, start=graph.node_list()[1])
    dfs_tree.plot(override_color="red")


    new_tree = dfs_tree.clone()
    depth_counter = Counter()
    for node in new_tree.nodes:
        depth = int(node.name[-2])  # HACK
        depth_counter[depth] += 1
        print(f"{node.name=} {depth=}")
        node.point = Point(
            x=depth_counter[depth] / 3,
            y=0.9 - depth / 5
        )
    fig, ax = new_tree.plot(
        show=False,
        override_color="black",
    )
    node_a = new_tree.get("A [d=0]")
    node_b = new_tree.get("B [d=2]")
    node_d = new_tree.get("D [d=2]")
    node_f = new_tree.get("F [d=4]")
    ax.plot(
        [node_a.point.x, 0.25, node_b.point.x],
        [node_a.point.y, 0.7, node_b.point.y],
        color="red",

    )
    ax.plot(
        [node_d.point.x, node_f.point.x],
        [node_d.point.y, node_f.point.y],
        color="red",
    )
    ax.set_title("")
    plt.show()
    pass
