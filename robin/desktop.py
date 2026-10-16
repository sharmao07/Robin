import sys
import threading

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QStackedWidget,
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
from robin.brain.model_manager import list_installed_models
from robin.brain.memory import (
    initialize_memory,
    load_recent_messages,
    save_turn,
)
from robin.tools.apps import open_app
from robin.tools.permissions import authorize
from robin.tools.router import parse_app_request
from robin.ui.activity import ActivityDialog
from robin.brain.personality import load_personality
from robin.ui.settings import PersonalitySettingsDialog
from robin.ui.theme import APP_STYLESHEET
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

    def request_stop(self):
        self.stop_event.set()

    def __init__(self, model, client, messages):
        super().__init__()
        self.model = model
        self.client = client
        self.stop_event = threading.Event()
        self.messages = [dict(message) for message in messages]

    def run(self):
        answer = []

        try:
            for kind, chunk in stream_reply(
            self.model,
            self.client,
            self.messages,
            stop_event=self.stop_event,
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
        self.messages = [{"role": "system", "content": load_personality()}]
        self.messages.extend(load_recent_messages(limit=8))
        self.active_prompt = ""
        self.display_kind = None
        self.thinking_displayed = 0
        self.thinking_limit = 300
        self.thinking_notice_shown = False
        self.nav_buttons = {}
        self.feature_pages = {}

        self.setWindowTitle("Robin — Your Local AI")
        self.resize(1240, 800)
        self.setMinimumSize(980, 640)

        shell = QWidget()
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(228)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 22, 16, 18)
        sidebar_layout.setSpacing(8)

        brand_row = QHBoxLayout()
        brand_mark = QLabel("R")
        brand_mark.setObjectName("BrandMark")
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(40, 40)
        brand_text = QVBoxLayout()
        brand_name = QLabel("robin")
        brand_name.setObjectName("BrandName")
        brand_caption = QLabel("PERSONAL AI")
        brand_caption.setObjectName("Eyebrow")
        brand_text.addWidget(brand_name)
        brand_text.addWidget(brand_caption)
        brand_row.addWidget(brand_mark)
        brand_row.addLayout(brand_text, 1)
        sidebar_layout.addLayout(brand_row)
        sidebar_layout.addSpacing(24)

        nav_label = QLabel("WORKSPACE")
        nav_label.setObjectName("Eyebrow")
        sidebar_layout.addWidget(nav_label)

        nav_items = [
            ("chat", "◈", "Chat"),
            ("knowledge", "▤", "Knowledge"),
            ("proactive", "✧", "Proactive"),
            ("sidekick", "⌘", "Sidekick"),
            ("quick", "↗", "Quick Actions"),
            ("background", "◷", "Background"),
        ]
        for key, glyph, label in nav_items:
            button = QPushButton(f"{glyph}    {label}")
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, page=key: self.show_page(page))
            self.nav_buttons[key] = button
            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch(1)
        privacy_card = QFrame()
        privacy_card.setObjectName("PrivacyCard")
        privacy_layout = QVBoxLayout(privacy_card)
        privacy_layout.setContentsMargins(12, 12, 12, 12)
        privacy_title = QLabel("●  LOCAL & PRIVATE")
        privacy_title.setObjectName("PrivacyTitle")
        privacy_text = QLabel("Your prompts stay on this Mac when using the local model.")
        privacy_text.setObjectName("MutedText")
        privacy_text.setWordWrap(True)
        privacy_layout.addWidget(privacy_title)
        privacy_layout.addWidget(privacy_text)
        sidebar_layout.addWidget(privacy_card)

        sidebar_footer = QHBoxLayout()
        self.settings_button = QPushButton("⚙  Settings")
        self.settings_button.setObjectName("SidebarUtility")
        self.settings_button.clicked.connect(self.open_personality_settings)
        self.activity_button = QPushButton("≡  Activity")
        self.activity_button.setObjectName("SidebarUtility")
        self.activity_button.clicked.connect(self.open_activity_viewer)
        sidebar_footer.addWidget(self.settings_button)
        sidebar_footer.addWidget(self.activity_button)
        sidebar_layout.addLayout(sidebar_footer)

        # Main workspace
        workspace = QWidget()
        workspace.setObjectName("Workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(28, 22, 28, 22)
        workspace_layout.setSpacing(18)

        topbar = QHBoxLayout()
        page_heading = QVBoxLayout()
        self.page_title = QLabel("Chat")
        self.page_title.setObjectName("PageTitle")
        self.page_subtitle = QLabel("A private assistant that works on your Mac.")
        self.page_subtitle.setObjectName("MutedText")
        page_heading.addWidget(self.page_title)
        page_heading.addWidget(self.page_subtitle)
        topbar.addLayout(page_heading, 1)

        self.status_pill = QLabel("●  LOCAL MODEL")
        self.status_pill.setObjectName("StatusPill")
        topbar.addWidget(self.status_pill)
        topbar.addSpacing(8)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(170)
        self.model_combo.setToolTip("Choose an installed local model")
        self.model_combo.currentTextChanged.connect(self.change_model)
        topbar.addWidget(self.model_combo)
        workspace_layout.addLayout(topbar)

        separator = QFrame()
        separator.setObjectName("Separator")
        separator.setFixedHeight(1)
        workspace_layout.addWidget(separator)

        self.page_stack = QStackedWidget()

        # Chat page: retain Robin's existing streaming chat handlers.
        chat_page = QWidget()
        chat_layout = QVBoxLayout(chat_page)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(14)

        welcome = QFrame()
        welcome.setObjectName("WelcomeCard")
        welcome_layout = QVBoxLayout(welcome)
        welcome_layout.setContentsMargins(20, 16, 20, 16)
        welcome_title = QLabel("What can we work on?")
        welcome_title.setObjectName("WelcomeTitle")
        welcome_copy = QLabel("Ask a question, build something, or choose a workspace tool from the sidebar.")
        welcome_copy.setObjectName("MutedText")
        welcome_copy.setWordWrap(True)
        welcome_layout.addWidget(welcome_title)
        welcome_layout.addWidget(welcome_copy)
        chat_layout.addWidget(welcome)

        self.transcript = QTextEdit()
        self.transcript.setObjectName("Transcript")
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("Your conversation will appear here…")
        chat_layout.addWidget(self.transcript, 1)

        composer = QFrame()
        composer.setObjectName("Composer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(12, 10, 12, 10)
        composer_layout.setSpacing(8)
        self.input = QLineEdit()
        self.input.setObjectName("PromptInput")
        self.input.setPlaceholderText("Message Robin…  (Return to send)")
        self.input.returnPressed.connect(self.send_message)
        composer_layout.addWidget(self.input)

        composer_actions = QHBoxLayout()
        composer_hint = QLabel("LOCAL INFERENCE  ·  YOU CONTROL ACTIONS")
        composer_hint.setObjectName("Eyebrow")
        composer_actions.addWidget(composer_hint, 1)
        self.new_chat_button = QPushButton("＋ New chat")
        self.new_chat_button.setObjectName("SecondaryButton")
        self.new_chat_button.clicked.connect(self.new_chat)
        composer_actions.addWidget(self.new_chat_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("SecondaryButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_generation)
        composer_actions.addWidget(self.stop_button)
        self.send_button = QPushButton("Send  ↗")
        self.send_button.setObjectName("PrimaryButton")
        self.send_button.clicked.connect(self.send_message)
        composer_actions.addWidget(self.send_button)
        composer_layout.addLayout(composer_actions)
        chat_layout.addWidget(composer)
        self.page_stack.addWidget(chat_page)
        self.feature_pages["chat"] = 0

        # Honest workspace landing pages; each is wired to a separate implementation milestone.
        feature_content = {
            "knowledge": (
                "Knowledge",
                "Build a private, searchable library from files you choose.",
                ["Import documents and notes", "Index text locally", "Review sources and remove them"],
                "Next: local file ingestion and search",
            ),
            "proactive": (
                "Proactive",
                "Turn context into suggestions you can review before acting.",
                ["Suggested next steps", "Drafts awaiting approval", "Reminders you control"],
                "Next: local task queue and reminder scheduler",
            ),
            "sidekick": (
                "Sidekick",
                "Let Robin assist with supported Mac tasks while you stay in control.",
                ["Approved application actions", "Visible task progress", "Stop and approval controls"],
                "Next: safe computer-use workflow",
            ),
            "quick": (
                "Quick Actions",
                "Reach Robin from a focused, lightweight action panel.",
                ["Draft text for review", "Copy prepared replies", "Keyboard-first workflows"],
                "Next: quick-action panel and macOS shortcut",
            ),
            "background": (
                "Background Intelligence",
                "Run small, inspectable jobs on a schedule you configure.",
                ["Scheduled local jobs", "Job status and history", "Pause and resume controls"],
                "Next: persistent scheduler and job worker",
            ),
        }
        for key, (title, description, bullets, status) in feature_content.items():
            page = self._make_feature_page(title, description, bullets, status)
            self.feature_pages[key] = self.page_stack.addWidget(page)

        workspace_layout.addWidget(self.page_stack, 1)
        shell_layout.addWidget(sidebar)
        shell_layout.addWidget(workspace, 1)
        self.setCentralWidget(shell)

        self.refresh_models()
        self.show_page("chat")
        self.append_text("Robin > I'm ready. How can I help?\n\n", GREEN)

    def _make_feature_page(self, title, description, bullets, status):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("FeatureHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(26, 24, 26, 24)
        eyebrow = QLabel("ROBIN WORKSPACE")
        eyebrow.setObjectName("Eyebrow")
        heading = QLabel(title)
        heading.setObjectName("FeatureTitle")
        copy = QLabel(description)
        copy.setObjectName("MutedText")
        copy.setWordWrap(True)
        hero_layout.addWidget(eyebrow)
        hero_layout.addWidget(heading)
        hero_layout.addWidget(copy)
        layout.addWidget(hero)

        section_title = QLabel("DESIGNED FOR YOUR WORKFLOW")
        section_title.setObjectName("Eyebrow")
        layout.addWidget(section_title)
        for index, bullet in enumerate(bullets, 1):
            row = QFrame()
            row.setObjectName("FeatureRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(16, 14, 16, 14)
            number = QLabel(f"0{index}")
            number.setObjectName("FeatureNumber")
            label = QLabel(bullet)
            label.setObjectName("FeatureRowText")
            row_layout.addWidget(number)
            row_layout.addWidget(label, 1)
            layout.addWidget(row)

        notice = QLabel(status)
        notice.setObjectName("MutedText")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        layout.addStretch(1)
        return page

    def show_page(self, key):
        if key not in self.feature_pages:
            return
        self.page_stack.setCurrentIndex(self.feature_pages[key])
        titles = {
            "chat": ("Chat", "A private assistant that works on your Mac."),
            "knowledge": ("Knowledge", "Your files, indexed locally and under your control."),
            "proactive": ("Proactive", "Suggestions first. Actions only with your approval."),
            "sidekick": ("Sidekick", "Computer assistance with visible, permission-gated actions."),
            "quick": ("Quick Actions", "Fast workflows for drafting and focused tasks."),
            "background": ("Background Intelligence", "Scheduled work with status and controls."),
        }
        title, subtitle = titles[key]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)
        for name, button in self.nav_buttons.items():
            button.setChecked(name == key)

    def stop_generation(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.stop_button.setEnabled(False)
            self.append_text("\nRobin > Stopping response...\n", GRAY)

    def append_text(self, text, color):
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(text, fmt)

        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def refresh_models(self):
        self.model_combo.blockSignals(True)
        try:
            names = list_installed_models(self.client)

            if self.model not in names:
                names.insert(0, self.model)

            self.model_combo.clear()
            self.model_combo.addItems(names)
            self.model_combo.setCurrentText(self.model)

        except Exception as error:
            self.model_combo.clear()
            self.model_combo.addItem(self.model)
            self.append_text(
                f"Robin > Could not list local models: {error}\\n\\n",
                GRAY,
            )
        finally:
            self.model_combo.blockSignals(False)

    def change_model(self, name):
        if not name or name == self.model:
            return

        if self.worker is not None and self.worker.isRunning():
            return

        self.model = name
        self.append_text(
            f"Robin > Active model changed to {name}.\\n\\n",
            GREEN,
        )

    def open_activity_viewer(self):
        dialog = ActivityDialog(self)
        dialog.exec()

    def open_personality_settings(self):
        dialog = PersonalitySettingsDialog(self)

        if dialog.exec():
            self.messages[0] = {
                "role": "system",
                "content": load_personality(),
            }
            self.append_text(
                "Robin > Personality updated.\\n\\n",
                GREEN,
            )

    def new_chat(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Robin is responding",
                "Wait for the current response to finish first.",
            )
            return

        self.messages = [{
            "role": "system",
            "content": load_personality(),
        }]

        self.transcript.clear()
        self.append_text(
            "Robin > New conversation started. How can I help?\\n\\n",
            GREEN,
        )

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
        self.stop_button.setEnabled(False)
        self.worker.finished.connect(self.on_worker_finished)
        self.stop_button.setEnabled(True)
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

        if self.runtime is not None:
            if hasattr(self.runtime, "shutdown"):
                self.runtime.shutdown()
            elif hasattr(self.runtime, "stop_if_started"):
                self.runtime.stop_if_started()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    app.setApplicationName("Robin")

    try:
        from robin.brain.providers import get_provider
        provider = get_provider()
        provider.initialize()
        model = provider.default_model
        initialize_memory()
        runtime = getattr(provider, "runtime", None)
    except Exception as error:
        QMessageBox.critical(
            None,
            "Robin could not start",
            f"{error}\n\n"
            "Please ensure the local GGUF model is downloaded before starting Robin offline.",
        )
        return 1

    window = RobinWindow(model, provider, runtime)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
