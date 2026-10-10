from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from robin.brain.personality import (
    DEFAULT_PERSONALITY,
    PERSONALITY_FILE,
)


class PersonalitySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Robin Personality Settings")
        self.resize(560, 420)

        layout = QVBoxLayout(self)

        title = QLabel("Customize your Robin")
        title.setStyleSheet(
            "font-size: 20px; font-weight: bold;"
        )

        description = QLabel(
            "Describe Robin's personality, tone, response length, "
            "and preferred explanation style."
        )
        description.setWordWrap(True)

        self.editor = QTextEdit()
        self.editor.setPlainText(
            PERSONALITY_FILE.read_text(encoding="utf-8")
            if PERSONALITY_FILE.exists()
            else DEFAULT_PERSONALITY
        )

        buttons = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")

        save_button.clicked.connect(self.save_settings)
        cancel_button.clicked.connect(self.reject)

        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.editor, 1)
        layout.addLayout(buttons)

    def save_settings(self):
        preferences = self.editor.toPlainText().strip()

        if not preferences:
            QMessageBox.warning(
                self,
                "Empty personality",
                "Enter some personality instructions before saving.",
            )
            return

        PERSONALITY_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        PERSONALITY_FILE.write_text(
            preferences + "\n",
            encoding="utf-8",
        )

        self.accept()
