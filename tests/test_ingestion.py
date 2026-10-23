"""Initial-release behavior checks. All files and databases use temporary directories."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from opsdesk.config import Settings
from opsdesk.ingestion import import_dataset


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = Settings(self.root / "data")

    def count(self, table):
        self.assertIn(table, {"devices", "events", "data_issues", "imports"})
        with sqlite3.connect(self.settings.db_path) as db:
            return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def modified_input(self, events=None, devices=None):
        directory = self.root / "input"
        directory.mkdir(exist_ok=True)
        for filename, records in (("devices.json", devices), ("events.json", events)):
            if records is None:
                payload = (EXAMPLES / "normal" / filename).read_bytes()
            else:
                payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
            (directory / filename).write_bytes(payload)
        return directory

    def events(self):
        return json.loads((EXAMPLES / "normal" / "events.json").read_text(encoding="utf-8"))

    def test_normal_import_and_byte_exact_archive(self):
        result = import_dataset(EXAMPLES / "normal", self.settings)
        self.assertEqual((result["accepted_devices"], result["accepted_events"]), (3, 6))
        self.assertEqual(result["rejected_records"], 0)
        self.assertEqual((self.count("devices"), self.count("events")), (3, 6))
        archive = self.settings.data_dir / "imports" / result["batch_id"]
        for filename in ("devices.json", "events.json"):
            self.assertEqual((archive / filename).read_bytes(), (EXAMPLES / "normal" / filename).read_bytes())
        self.assertEqual(list((self.settings.data_dir / "quarantine").glob("*.json")), [])

    def test_same_batch_is_idempotent(self):
        first = import_dataset(EXAMPLES / "normal", self.settings)
        second = import_dataset(EXAMPLES / "normal", self.settings)
        self.assertEqual(first["batch_id"], second["batch_id"])
        self.assertTrue(second["already_imported"])
        self.assertEqual(self.count("events"), 6)
        self.assertEqual(self.count("imports"), 1)

    def test_invalid_timestamp_retains_original_record(self):
        result = import_dataset(EXAMPLES / "invalid_event", self.settings)
        self.assertEqual((result["accepted_events"], result["rejected_records"]), (5, 1))
        issue_file = next((self.settings.data_dir / "quarantine").glob("*.json"))
        issue = json.loads(issue_file.read_text(encoding="utf-8"))
        self.assertEqual(issue["raw_record"]["occurred_at"], "INVALID_TIMESTAMP")
        self.assertEqual(issue["source_index"], 1)
        self.assertEqual(issue["batch_id"], result["batch_id"])

    def test_unknown_device_is_rejected_without_losing_other_events(self):
        events = self.events()
        events[0]["device_id"] = "UNKNOWN"
        result = import_dataset(self.modified_input(events=events), self.settings)
        self.assertEqual((result["accepted_events"], result["rejected_records"]), (5, 1))
        self.assertEqual(self.count("data_issues"), 1)

    def test_conflicting_event_is_quarantined_original_retained(self):
        import_dataset(EXAMPLES / "normal", self.settings)
        events = self.events()
        events[1]["message"] = "conflicting incoming message"
        result = import_dataset(self.modified_input(events=events), self.settings)
        self.assertEqual(result["rejected_records"], 1)
        self.assertEqual(self.count("events"), 6)
        with sqlite3.connect(self.settings.db_path) as db:
            message = db.execute("SELECT message FROM events WHERE event_id = ?", ("EV-002",)).fetchone()[0]
            raw = db.execute("SELECT raw_json FROM data_issues").fetchone()[0]
        self.assertNotEqual(message, "conflicting incoming message")
        self.assertEqual(json.loads(raw)["message"], "conflicting incoming message")

    def test_conflicting_device_blocks_its_incoming_events(self):
        import_dataset(EXAMPLES / "normal", self.settings)
        devices = json.loads((EXAMPLES / "normal" / "devices.json").read_text(encoding="utf-8"))
        devices[0]["model"] = "CONFLICTING-MODEL"
        event = self.events()[1]
        event["event_id"] = "NEW-EVENT"
        result = import_dataset(self.modified_input(events=[event], devices=devices), self.settings)
        self.assertEqual(result["rejected_records"], 2)
        self.assertEqual(result["accepted_events"], 0)
        self.assertEqual(self.count("events"), 6)

    def test_file_reformatting_does_not_duplicate_records(self):
        import_dataset(EXAMPLES / "normal", self.settings)
        result = import_dataset(self.modified_input(events=self.events()), self.settings)
        self.assertFalse(result["already_imported"])
        self.assertEqual(result["skipped_duplicates"], 9)
        self.assertEqual(self.count("events"), 6)

    def test_invalid_json_is_archived_without_partial_rows(self):
        directory = self.modified_input()
        raw = b'[{"event_id": "unfinished"'
        (directory / "events.json").write_bytes(raw)
        with self.assertRaises(ValueError):
            import_dataset(directory, self.settings)
        self.assertEqual(self.count("devices"), 0)
        self.assertEqual(self.count("imports"), 0)
        archived = next((self.settings.data_dir / "imports").glob("*/events.json"))
        self.assertEqual(archived.read_bytes(), raw)

    def test_timezone_is_required_and_valid_time_is_normalized(self):
        events = self.events()
        events[1]["occurred_at"] = "2026-10-19T07:10:00"
        result = import_dataset(self.modified_input(events=events), self.settings)
        self.assertEqual(result["rejected_records"], 1)
        with sqlite3.connect(self.settings.db_path) as db:
            occurred_at = db.execute("SELECT occurred_at FROM events WHERE event_id = ?", ("EV-001",)).fetchone()[0]
        self.assertEqual(occurred_at, "2026-10-18T22:30:00.000000+00:00")

    def test_reimport_restores_missing_issue_export(self):
        import_dataset(EXAMPLES / "invalid_event", self.settings)
        issue_file = next((self.settings.data_dir / "quarantine").glob("*.json"))
        issue_file.unlink()
        result = import_dataset(EXAMPLES / "invalid_event", self.settings)
        self.assertTrue(result["already_imported"])
        self.assertTrue(issue_file.is_file())
        self.assertEqual(self.count("data_issues"), 1)

    def test_additional_source_fields_are_retained(self):
        events = self.events()
        events[0]["vendor_payload"] = {"sequence": 42, "notes": ["a", "b"]}
        import_dataset(self.modified_input(events=events), self.settings)
        with sqlite3.connect(self.settings.db_path) as db:
            raw = db.execute("SELECT raw_json FROM events WHERE event_id = ?", ("EV-001",)).fetchone()[0]
        self.assertEqual(json.loads(raw)["vendor_payload"]["sequence"], 42)

    def test_cli_success_and_missing_input_failure(self):
        common = [sys.executable, "-m", "opsdesk.cli", "--data-dir", str(self.settings.data_dir)]
        success = subprocess.run([*common, "import", str(EXAMPLES / "normal")], capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(json.loads(success.stdout)["accepted_events"], 6)
        failure = subprocess.run([*common, "import", str(self.root / "missing")], capture_output=True, text=True)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("失败", failure.stderr)


if __name__ == "__main__":
    unittest.main()
