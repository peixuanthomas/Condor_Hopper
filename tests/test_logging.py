import csv
from datetime import datetime, timezone

from condor_hopper.flight_logging import FlightLogger, TELEMETRY_FIELDS


def test_logger_uses_date_and_project_subfolder(tmp_path):
    now = datetime(2026, 8, 3, 12, 34, 56, tzinfo=timezone.utc)
    with FlightLogger(tmp_path, now=now) as logger:
        logger.telemetry({"phase": "preflight", "ut": 1.0})
        logger.event("test-event", value=3)

    assert logger.directory.name == "2026-08-03_Condor_Hopper"
    assert logger.telemetry_path.parent == logger.directory
    with logger.telemetry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == TELEMETRY_FIELDS
    assert rows[0]["phase"] == "preflight"
    assert logger.events_path.read_text(encoding="utf-8").count("test-event") == 1

