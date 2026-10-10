import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from robin.brain.audit import record_event
from robin.brain.local_llm import load_brain, stream_reply
from robin.brain.memory import (
    initialize_memory,
    load_recent_messages,
    save_turn,
)
from robin.brain.runtime import OllamaRuntime
from robin.tools.apps import open_app
from robin.tools.permissions import authorize
from robin.tools.router import parse_app_request
from robin.brain.personality import load_personality


SYSTEM_PROMPT = load_personality()


CYAN = "#67E8F9"
GREEN = "#4ADE80"
GRAY = "#9CA3AF"
RED = "#F87171"


class ReplyWorker(QThread):
    response_chunk = Signal(str, str)
    response_ready = Signal(str)
    response_error = Signal(str)

    def __init__(self, model, client, messages):
        super().__init__()
        self.model = model
        self.client = client
        self.messages = [dict(message) for message in messages]

    def run(self):
        answer = []

        try:
            for kind, chunk in stream_reply(
                self.model,
                self.client,
                self.messages,
            ):
                self.response_chunk.emit(kind, chunk)

                if kind == "answer":
                    answer.append(chunk)

            self.response_ready.emit("".join(answer).strip())

        except Exception as error:
            self.response_error.emit(str(error))


class RobinWindow(QMainWindow):
    def __init__(self, model, client, runtime):
        super().__init__()

        self.model = model
        self.client = client
        self.runtime = runtime
        self.worker = None

        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.messages.extend(load_recent_messages(limit=8))

        self.active_prompt = ""
        self.display_kind = None
        self.thinking_displayed = 0
        self.thinking_limit = 300
        self.thinking_notice_shown = False

        self.setWindowTitle("Robin — Local AI")
        self.resize(920, 680)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        heading = QLabel("ROBIN  /  LOCAL AI")
        heading.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #E879F9;"
        )

        subtitle = QLabel(
            "Private, local chat • Ollama runtime managed by Robin"
        )
        subtitle.setStyleSheet("color: #9CA3AF;")

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setStyleSheet(
            "QTextEdit { background: #111827; color: #E5E7EB; "
            "border: 1px solid #374151; border-radius: 10px; "
            "padding: 12px; font-size: 14px; }"
        )

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask Robin anything...")
        self.input.setStyleSheet(
            "QLineEdit { background: #1F2937; color: #F9FAFB; "
            "border: 1px solid #4B5563; border-radius: 8px; "
            "padding: 12px; font-size: 14px; }"
        )
        self.input.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setStyleSheet(
            "QPushButton { background: #7C3AED; color: white; "
            "border: none; border-radius: 8px; padding: 12px 20px; "
            "font-weight: bold; } "
            "QPushButton:hover { background: #6D28D9; }"
        )

        input_row = QHBoxLayout()
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.send_button)

        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addWidget(self.transcript, 1)
        layout.addLayout(input_row)

        self.setCentralWidget(root)
        self.setStyleSheet("QMainWindow { background: #0B1120; }")

        self.append_text(
            "Robin > I'm ready. How can I help?\n\n",
            GREEN,
        )

    def append_text(self, text, color):
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(text, fmt)

        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def send_message(self):
        if self.worker is not None and self.worker.isRunning():
            return

        prompt = self.input.text().strip()
        if not prompt:
            return

        self.input.clear()
        self.append_text(f"You > {prompt}\n", CYAN)

        # Explicit app requests use Robin's existing gated tools.
        app_name = parse_app_request(prompt)

        if app_name is not None:
            try:
                if authorize("open_app", app_name):
                    result = open_app(app_name)
                    record_event("open_app", app_name, result)
                    self.append_text(f"Robin > {result}\n\n", GREEN)
                else:
                    record_event("open_app", app_name, "blocked")
                    self.append_text(
                        "Robin > This action is not permitted.\n\n",
                        GREEN,
                    )
            except Exception as error:
                self.append_text(f"Robin > {error}\n\n", RED)

            return

        self.messages.append({"role": "user", "content": prompt})
        self.active_prompt = prompt
        self.display_kind = None
        self.thinking_displayed = 0
        self.thinking_notice_shown = False

        self.input.setEnabled(False)
        self.send_button.setEnabled(False)

        self.worker = ReplyWorker(
            self.model,
            self.client,
            self.messages,
        )
        self.worker.response_chunk.connect(self.on_response_chunk)
        self.worker.response_ready.connect(self.on_response_ready)
        self.worker.response_error.connect(self.on_response_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_response_chunk(self, kind, chunk):
        if not chunk:
            return

        if kind == "thinking":
            if self.display_kind != "thinking":
                if self.display_kind is not None:
                    self.append_text("\n", GRAY)

                self.append_text("Thinking > ", GRAY)
                self.display_kind = "thinking"

            remaining = max(
                0,
                self.thinking_limit - self.thinking_displayed,
            )
            visible = chunk[:remaining]

            if visible:
                self.append_text(visible, GRAY)
                self.thinking_displayed += len(visible)

            if (
                len(visible) < len(chunk)
                and not self.thinking_notice_shown
            ):
                self.append_text(
                    " ... [thinking display shortened]",
                    GRAY,
                )
                self.thinking_notice_shown = True

            return

        if self.display_kind != "answer":
            if self.display_kind is not None:
                self.append_text("\n", GRAY)

            self.append_text("Robin > ", GREEN)
            self.display_kind = "answer"

        self.append_text(chunk, GREEN)

    def on_response_ready(self, answer):
        if self.display_kind is not None:
            self.append_text("\n\n", GREEN)

        if not answer:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()

            self.append_text(
                "Robin > I couldn't produce a final answer. Try again.\n\n",
                RED,
            )
            return

        self.messages.append({
            "role": "assistant",
            "content": answer,
        })

        try:
            save_turn(self.active_prompt, answer)
        except Exception as error:
            self.append_text(
                f"\nMemory warning: {error}\n",
                RED,
            )

    def on_response_error(self, error):
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()

        if self.display_kind is not None:
            self.append_text("\n", RED)

        self.append_text(f"Robin > I couldn't respond: {error}\n\n", RED)

    def on_worker_finished(self):
        self.worker = None
        self.input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input.setFocus()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Robin is working",
                "Wait for Robin's current response to finish before closing.",
            )
            event.ignore()
            return

        self.runtime.stop_if_started()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Robin")

    runtime = OllamaRuntime()

    try:
        runtime.ensure_running()
        model, client = load_brain()
        initialize_memory()
    except Exception as error:
        runtime.stop_if_started()

        QMessageBox.critical(
            None,
            "Robin could not start",
            f"{error}\n\n"
            "If the model is missing, download qwen3:4b once "
            "before using Robin offline.",
        )
        return 1

    window = RobinWindow(model, client, runtime)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
