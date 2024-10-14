import plotly.graph_objects as go
import networkx as nx

import parsed_snapshot


_IMAGE_WIDTH = 922
_IMAGE_HEIGHT = 573
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
positions = {
    k: (v[0] / _IMAGE_WIDTH, 1 - v[1] / _IMAGE_HEIGHT) for k, v in positions.items()
}


G = nx.random_geometric_graph(200, 0.125, seed=40351)

snap = parsed_snapshot.ParsedSnapshot("./logs")
print(snap)

G = snap.networkx_graph()
for node in G.nodes:
    G.nodes[node]["pos"] = positions[node]

# exit()


print(G)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]["pos"]
    x1, y1 = G.nodes[edge[1]]["pos"]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

from rich.pretty import pprint

pprint(edge_x)
# exit()

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line={"width": 2.5, "color": "#888"},
    hoverinfo="none",
    mode="lines",
)

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]["pos"]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    hoverinfo="text",
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale="YlGnBu",
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15, title="Node Connections", xanchor="left", titleside="right"
        ),
        line_width=2,
    ),
)


node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(f"# of connections: " + str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text


fig = go.Figure(
    data=[
        edge_trace,
        node_trace,
    ],
    layout=go.Layout(
        title="Network graph made with Python",
        titlefont_size=16,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Python code: <a href='https://plotly.com/python/network-graphs/'> https://plotly.com/python/network-graphs/</a>",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),
)
fig.show()
