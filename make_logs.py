"""
Make some simplified Rajant Breadcrumb logs
"""

from collections import defaultdict
import datetime
import heapq
from pathlib import Path
import random

from rich.pretty import pprint

from parsed_log import InstameshRoutingTableEntry


random.seed(40351)

# All connected nodes from the powerpoint example
peer_pairs = [
    "AB",
    "AC",
    "BG",
    "CD",
    "CG",
    "CH",
    "DE",
    "DH",
    "DI",
    "DI",
    "EF",
    "EI",
    "EJ",
    "EN",
    "FJ",
    "GK",
    "HI",
    "HL",
    "IL",
    "IM",
    "IN",
    "JN",
    "JO",
    "KL",
    "LM",
    "LP",
    "MN",
    "MP",
    "MQ",
    "MR",
    "NO",
    "NS",
    "OS",
    "RS",
]


edge_list = [(peer_pair[0], peer_pair[1], 1) for peer_pair in peer_pairs]
"""Format is `(source, destination, weight)`"""


def create_adjacency_list(
    edge_list: list[tuple[str, str, int]],
) -> dict[str, dict[str, int]]:
    """Create the adjacency "list", which is a `dict` where keys are source nodes, and values
    are `dict`s of destinations to weights

    NOTE: Assumes bidirectional connectivity of all edges!

    Args:
        edge_list (list[tuple[str, str, int]]): list of `(source, destination, weight)`

    Returns:
        dict[str, dict[str, int]]: adjacency "list"
    """
    adjacency_list = defaultdict(dict)
    for source, destination, weight in edge_list:
        adjacency_list[source][destination] = weight
        adjacency_list[destination][source] = weight
    return dict(adjacency_list)


adjacency_list = create_adjacency_list(edge_list)
print(adjacency_list)


# Just copied from
# https://datagy.io/dijkstras-algorithm-python/#Implementing_Dijkstras_Algorithm_in_Python
# With "next hop" logic bolted on
def dijkstra(
    adj_list: dict[str, dict[str, int]],
    start_node: str,
) -> dict[str, InstameshRoutingTableEntry]:
    paths: defaultdict[str, list[str]] = defaultdict(list)
    distances = {node: float("inf") for node in adj_list}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in adj_list[current_node].items():
            total_cost = current_distance + weight
            if total_cost < distances[neighbor]:
                distances[neighbor] = total_cost
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (total_cost, neighbor))
    rv = {}
    for destination_node in distances:
        path = paths[destination_node]
        if path:
            next_hop_node = path[0]
            next_hop_cost = distances[next_hop_node]
            reachable = True
        else:
            next_hop_node = None
            next_hop_cost = None
            reachable = False
        total_cost = distances[destination_node]
        rv[destination_node] = InstameshRoutingTableEntry(
            path=path,
            next_hop_node=next_hop_node,
            next_hop_cost=next_hop_cost,
            total_cost=total_cost,
            reachable=reachable,
        )
    return rv


routing_table = dijkstra(adjacency_list, "A")
pprint(routing_table)


log_folder_path = Path(__file__).parent / "logs"
log_folder_path.mkdir(parents=True, exist_ok=True)

for letter in adjacency_list:
    # letter = chr(ord("A") + encap)
    log_path = log_folder_path / f"{letter}.log"

    # Add some random time offsets
    timestamp = datetime.datetime.fromisoformat(
        "2024-09-23T12:00:00"
    ) + datetime.timedelta(seconds=random.randrange(-600, 600))

    adjacencies = [adjacency for adjacency in peer_pairs if (letter in adjacency)]

    peers = sorted((adjacency.replace(letter, "") for adjacency in adjacencies))

    print(f"{letter=} {log_path=} {adjacencies=} {peers=}")

    with open(log_path, "w", encoding="utf-8") as file:
        file.write(f"#  breadcrumb log time\n")
        file.write(f"{timestamp.isoformat(sep=' ')}\n")
        file.write(f"\n")

        file.write(f"#  breadcrumb serial number\n")
        file.write(f"{letter}\n")
        file.write(f"\n")

        file.write(f"#  instamesh neighbors\n")
        file.write(f"neighbor    cost\n")
        file.write(f"--------    ----\n")
        for peer in peers:
            file.write(f"{peer}           1\n")
        file.write(f"\n")

        file.write(f"#  instamesh routing table\n")
        file.write(f"Dest    cost    next hop    next hop cost\n")
        file.write(f"----    ----    --------    -------------\n")
        routing_table = dijkstra(adjacency_list, letter)
        for key in sorted(routing_table):
            value = routing_table[key]
            if not value.reachable:
                continue
            file.write(
                f"{key}       {value.total_cost}       {value.next_hop_node}           {value.next_hop_cost}\n"
            )
        file.write(f"\n")
        file.write(f"#  Miscellaneous\n")
        file.write(f"Some kind of section that we can't parse\n")
        file.write(f"in any meaningful way.\n")
