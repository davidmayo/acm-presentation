from pathlib import Path

from parsed_log import InstameshRoutingTableEntry, ParsedLog


class ParsedSnapshot:
    def __init__(self, path: Path):
        self.path = path
        self.log_paths = sorted(path.glob("*.log"))
        self.parsed_logs: dict[str, ParsedLog] = {}
        for log_path in self.log_paths:
            parsed_log = ParsedLog(log_path)
            serial = parsed_log.serial_number
            if not serial:
                continue
            self.parsed_logs[serial] = parsed_log

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
