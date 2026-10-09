import unittest

from robin.tools.permissions import authorize


class PermissionTests(unittest.TestCase):
    def test_open_app_is_allowed(self):
        self.assertTrue(authorize("open_app", "Safari"))

    def test_unknown_action_is_denied(self):
        self.assertFalse(
            authorize("unknown_action", "anything")
        )

    def test_delete_is_denied(self):
        self.assertFalse(
            authorize("delete_file", "important.txt")
        )

    def test_writing_requires_explicit_approval(self):
        self.assertTrue(
            authorize(
                "write_file",
                "notes.txt",
                lambda _: "YES",
            )
        )

        self.assertFalse(
            authorize(
                "write_file",
                "notes.txt",
                lambda _: "no",
            )
        )


if __name__ == "__main__":
    unittest.main()
