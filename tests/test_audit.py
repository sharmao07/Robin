import json
import tempfile
import unittest
from pathlib import Path

from robin.brain import audit


class AuditTests(unittest.TestCase):
    def test_record_event_writes_json(self):
        original_path = audit.AUDIT_PATH

        try:
            with tempfile.TemporaryDirectory() as directory:
                audit.AUDIT_PATH = (
                    Path(directory) / "activity.jsonl"
                )

                audit.record_event(
                    "open_app",
                    "calculator",
                    "success",
                )

                lines = audit.AUDIT_PATH.read_text().splitlines()
                self.assertEqual(len(lines), 1)

                event = json.loads(lines[0])
                self.assertEqual(event["action"], "open_app")
                self.assertEqual(event["target"], "calculator")
                self.assertEqual(event["outcome"], "success")
                self.assertIn("time", event)

        finally:
            audit.AUDIT_PATH = original_path


if __name__ == "__main__":
    unittest.main()
