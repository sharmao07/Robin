import json

from PySide6.QtWidgets import (
    QDialog,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from robin.brain.audit import AUDIT_PATH


class ActivityDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Robin Activity")
        self.resize(700, 450)

        layout = QVBoxLayout(self)
        self.viewer = QPlainTextEdit()
        self.viewer.setReadOnly(True)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)

        layout.addWidget(self.viewer, 1)
        layout.addWidget(refresh_button)

        self.refresh()

    def refresh(self):
        if not AUDIT_PATH.exists():
            self.viewer.setPlainText(
                "No activity has been recorded yet."
            )
            return

        entries = []

        for line in AUDIT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()[-200:]:
            try:
                event = json.loads(line)

                entries.append(
                    f"{event.get('time', '')} | "
                    f"{event.get('action', '')} | "
                    f"{event.get('target', '')} | "
                    f"{event.get('outcome', '')}"
                )
            except json.JSONDecodeError:
                continue

        self.viewer.setPlainText(
            "\n".join(entries) or "No activity has been recorded yet."
        )
