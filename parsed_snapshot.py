from pathlib import Path

from parsed_log import ParsedLog


class ParsedSnapshot:
    def __init__(self, path: Path):
        self.path = path
        


if __name__ == "__main__":
    path = Path(__file__).parent / "logs"
