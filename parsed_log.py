import dataclasses
import datetime
import functools
from pathlib import Path




@dataclasses.dataclass
class InstameshRoutingTableEntry:
    destination: str | None = None
    path: list[str] | None = dataclasses.field(default=None, repr=False)
    next_hop_node: str | None = None
    next_hop_cost: int | None = None
    total_cost: int | None = None
    reachable: bool | None = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass
class InstameshNeighborEntry:
    destination: str | None = None
    cost: int | None = None


@dataclasses.dataclass
class ParsedSection:
    title: str
    text: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        self.parse_section_data()

    def parse_section_data(self) -> None:
        ...


@dataclasses.dataclass
class ParsedSectionLogTime(ParsedSection):
    timestamp: datetime.datetime | None = dataclasses.field(default=None)
    def parse_section_data(self) -> None:
        self.timestamp = datetime.datetime.fromisoformat(self.text.strip())


@dataclasses.dataclass
class ParsedSectionSerialNumber(ParsedSection):
    serial: str | None = dataclasses.field(default=None)
    def parse_section_data(self) -> None:
        self.serial = self.text.strip()


@dataclasses.dataclass
class ParsedSectionInstameshNeighbors(ParsedSection):
    neighbors: dict[str, InstameshNeighborEntry] | None = dataclasses.field(default=None)
    def parse_section_data(self) -> None:
        self.neighbors = {}
        for line in self.text.splitlines():
            if not line or line.startswith("neighbor") or line.startswith("----"):
                continue
            destination_token, cost_token = line.split()
            self.neighbors[destination_token] = InstameshNeighborEntry(
                destination=destination_token,
                cost=int(cost_token)
            )


@dataclasses.dataclass
class ParsedSectionInstameshRoutingTable(ParsedSection):
    entries: dict[str, InstameshRoutingTableEntry] | None = dataclasses.field(default=None)

    def parse_section_data(self) -> None:
        self.entries = {}
        for line in self.text.splitlines():
            if not line or line.startswith("Dest") or line.startswith("----"):
                continue
            destination_token, cost_token, next_hop_token, next_hop_cost_token = line.split()
            self.entries[destination_token] = InstameshRoutingTableEntry(
                destination=destination_token,
                total_cost=int(cost_token),
                next_hop_cost=int(next_hop_cost_token),
                next_hop_node=next_hop_token,
                reachable=True,
            )
        pass


def parse_section(
    title: str,
    text: str,
) -> ParsedSection:
    cls = ParsedSection
    if title == "#  breadcrumb log time":
        cls = ParsedSectionLogTime
    elif title == "#  breadcrumb serial number":
        cls = ParsedSectionSerialNumber
    elif title == "#  instamesh neighbors":
        cls = ParsedSectionInstameshNeighbors
    elif title == "#  instamesh routing table":
        cls = ParsedSectionInstameshRoutingTable
    
    return cls(
        title=title,
        text=text,
    )


class ParsedLog:
    """
    A parsed breadcrumb log file.

    Parsed data is in the `.parsed_sections` dict, 
    """
    def __init__(self, path: Path) -> None:
        self.path = path
        self.raw_text = path.read_text()
        self.parsed_sections: dict[str, ParsedSection] = {}
        self._parse_all_sections()

    @functools.cached_property
    def serial_number(self) -> str | None:
        for value in self.parsed_sections.values():
            if isinstance(value, ParsedSectionSerialNumber):
                return value.serial
        return None
    
    @functools.cached_property
    def instamesh_neighbors(self) -> dict[str, InstameshNeighborEntry] | None:
        for value in self.parsed_sections.values():
            if isinstance(value, ParsedSectionInstameshNeighbors):
                return value.neighbors
        return None
    
    @functools.cached_property
    def instamesh_routing_table(self) -> dict[str, InstameshRoutingTableEntry] | None:
        for value in self.parsed_sections.values():
            if isinstance(value, ParsedSectionInstameshRoutingTable):
                return value.entries
        return None
    
    def _parse_all_sections(self):
        def finalize_section(section_lines: list[str]):
            if section_lines:
                title = section_lines[0]
                text = "\n".join(section_lines[1 : ])
                self.parsed_sections[title] = parse_section(
                    title=title,
                    text=text,
                )

        lines = self.raw_text.splitlines()
        section_lines = []
        while lines:
            current_line = lines.pop(0)
            if current_line.startswith("#  "):
                finalize_section(section_lines)
                section_lines = [current_line]
            else:
                section_lines.append(current_line)
        
        finalize_section(section_lines)


if __name__ == "__main__":
    from rich.pretty import pprint
    log_folder_path = Path(__file__).parent / "logs"
    log_file_paths = sorted(log_folder_path.glob("*.log"))
    log_file_path = log_file_paths[0]
    parsed_log = ParsedLog(log_file_path)
    pprint(parsed_log.path)
    pprint(parsed_log.parsed_sections)
    pprint(f"{parsed_log.serial_number=}")
    pprint(parsed_log.instamesh_neighbors)
    pprint(parsed_log.instamesh_routing_table)