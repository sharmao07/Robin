import unittest


class DesktopImportTests(unittest.TestCase):
    def test_desktop_window_can_be_imported(self):
        from robin.desktop import RobinWindow
        self.assertTrue(callable(RobinWindow))


if __name__ == "__main__":
    unittest.main()
