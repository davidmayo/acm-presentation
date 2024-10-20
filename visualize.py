import plotly.graph_objects as go
import networkx as nx

from PIL import Image
from rich.pretty import pprint

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


graph = nx.random_geometric_graph(200, 0.125, seed=40351)

snap = parsed_snapshot.ParsedSnapshot("./logs")
print(snap)

graph = snap.networkx_graph()
for node in graph.nodes:
    graph.nodes[node]["pos"] = positions[node]

# exit()


print(graph)

edge_x = []
edge_y = []
for edge in graph.edges():
    x0, y0 = graph.nodes[edge[0]]["pos"]
    x1, y1 = graph.nodes[edge[1]]["pos"]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)


pprint(edge_x)
# exit()


# fig = go.Figure()

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=go.scatter.Line(
        width=5,
        color="#0000aa",
    ),
    # line={"width": 1, "color": "#888888"},
    hoverinfo="none",
    mode="lines",
)
# fig.add_trace(edge_trace)

node_x = []
node_y = []
node_names = []
for node in graph.nodes():
    x, y = graph.nodes[node]["pos"]
    node_x.append(x)
    node_y.append(y)
    node_names.append(str(node) * 5)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    hoverinfo="text",
    text=node_names,
    textposition="bottom center",
    textfont=go.scatter.Textfont(
        color="#000000",
        size=20,
        family="Consolas,Courier New,monospace",
        weight="bold",
    ),
    marker=go.scatter.Marker(
        showscale=False,
        color="#00ffff",
        size=30,
        line_width=2,
    ),
)
# fig.add_trace(node_trace)


node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(graph.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(f"# of connections: " + str(len(adjacencies[1])))

# node_trace.marker.color = node_adjacencies
# node_trace.text = node_text


fig = go.Figure(
    data=[
        edge_trace,
        node_trace,
    ],
    layout=go.Layout(
        showlegend=False,
        margin={
            "b": 0,
            "l": 0,
            "r": 0,
            "t": 0,
        },
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
        },
    ),
)

# fig.add_layout_image(
#     {
#         "source": "/home/dmayo/dev/playground/acm-presentation/port.png",
#         "xref": "x",
#         "yref": "y",
#         "x": 0.5,
#         "y": 0.5,
#         "sizex": 1,
#         "sizey": 1,
#         "sizing": "stretch",
#         "opacity": 0.5,
#         "layer": "below",
#     }
# )

img = Image.open("port.png")
fig.add_layout_image(
    dict(
        # source="https://images.plot.ly/language-icons/api-home/python-logo.png",
        source=img,
        xref="x",
        yref="y",
        x=0,
        y=1,
        sizex=1,
        sizey=1,
        sizing="stretch",
        opacity=0.1,
        layer="below",
    )
)

# images = []
# fig.for_each_layout_image(lambda image: images.append(image))

# pprint(images)
# image: go.Image = images[0]


fig.update_layout(template="plotly_white")
fig.show()

# exit()

# fig = go.Figure(
#     data=[
#         edge_trace,
#         node_trace,
#     ],
#     layout=go.Layout(
#         # title="Network graph made with Python",
#         titlefont_size=16,
#         showlegend=False,
#         hovermode="closest",
#         margin=dict(b=20, l=5, r=5, t=40),
#         annotations=[
#             dict(
#                 text="Python code: <a href='https://plotly.com/python/network-graphs/'> https://plotly.com/python/network-graphs/</a>",
#                 showarrow=False,
#                 xref="paper",
#                 yref="paper",
#                 x=0.005,
#                 y=-0.002,
#             )
#         ],
#         xaxis=dict(
#             showgrid=False,
#             zeroline=False,
#             showticklabels=False,
#         ),
#         yaxis=dict(
#             showgrid=False,
#             zeroline=False,
#             showticklabels=False,
#         ),
#     ),
# )


# fig.show()
