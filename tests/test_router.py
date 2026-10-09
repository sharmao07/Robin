import unittest

from robin.tools.router import parse_app_request


class RouterTests(unittest.TestCase):
    def test_simple_request(self):
        self.assertEqual(
            parse_app_request("open safari"), "safari"
        )

    def test_robin_request(self):
        self.assertEqual(
            parse_app_request("Robin, please open Safari"),
            "safari",
        )

    def test_vscode_alias(self):
        self.assertEqual(
            parse_app_request("Could you launch VS Code?"),
            "vscode",
        )

    def test_notes_request(self):
        self.assertEqual(
            parse_app_request("Start Notes"), "notes"
        )

    def test_normal_question_is_not_an_app_request(self):
        self.assertIsNone(
            parse_app_request("Explain Python loops")
        )


if __name__ == "__main__":
    unittest.main()
