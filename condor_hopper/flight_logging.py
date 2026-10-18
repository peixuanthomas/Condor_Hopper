import csv
import json
from datetime import datetime
from pathlib import Path


class FlightLogger:
    def __init__(self, root):
        self.directory = Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.directory.mkdir(parents=True)
        self.file = (self.directory / "telemetry.csv").open("w", newline="", encoding="utf-8")
        self.writer = None

    def json(self, filename, value):
        (self.directory / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def row(self, value):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(value))
            self.writer.writeheader()
        self.writer.writerow(value)
        self.file.flush()

    def close(self):
        self.file.close()
