import importlib.util
import unittest


_HAS_PYSIDE6 = importlib.util.find_spec("PySide6") is not None


class DesktopImportTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_PYSIDE6, "PySide6 is not installed in the current environment")
    def test_desktop_window_can_be_imported(self):
        from robin.desktop import RobinWindow
        self.assertTrue(callable(RobinWindow))


if __name__ == "__main__":
    unittest.main()
