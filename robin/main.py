import sys
import time

from robin.brain.local_llm import load_brain, stream_reply
from robin.brain.memory import (
    initialize_memory,
    load_recent_messages,
    save_turn,
)
from robin.brain.audit import record_event
from robin.tools.apps import open_app
from robin.tools.permissions import authorize
from robin.tools.router import parse_app_request
from robin.brain.personality import load_personality


RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
GRAY = "\033[90m"

if not sys.stdout.isatty():
    RESET = CYAN = GREEN = MAGENTA = YELLOW = GRAY = ""


SYSTEM_PROMPT = load_personality()


def robin_says(message):
    print(f"{GREEN}Robin > {message}{RESET}\n")


def main():
    print(f"{MAGENTA}")
    print("==========================================")
    print("             ROBIN | LOCAL AI")
    print("==========================================")
    print(RESET)

    print(f"{YELLOW}Connecting to Robin's local AI...{RESET}")

    try:
        model, client = load_brain()
    except Exception as error:
        print(f"{YELLOW}Could not load Robin's AI: {error}{RESET}")
        return

    try:
        initialize_memory()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        messages.extend(load_recent_messages(limit=8))
    except Exception as error:
        print(f"{YELLOW}Could not initialize memory: {error}{RESET}")
        return

    robin_says("I'm ready. How can I help?")
    print(f"{YELLOW}Type 'help' for commands or 'exit' to quit.{RESET}\n")

    while True:
        try:
            print(f"{CYAN}You > ", end="", flush=True)
            command = input().strip()
            print(RESET, end="", flush=True)
        except (EOFError, KeyboardInterrupt):
            print()
            robin_says("Goodbye.")
            break

        if not command:
            continue

        lowered = command.lower()

        if lowered in ("exit", "quit"):
            robin_says("Goodbye.")
            break

        if lowered == "help":
            robin_says(
                "You can chat with me or ask me to open an approved app.\n"
                "Examples: open safari, launch vscode, start notes.\n"
                "Type exit to quit."
            )
            continue

        # Only explicit app-launch requests reach this tool.
        app_name = parse_app_request(command)

        if app_name is not None:
            try:
                if authorize("open_app", app_name):
                    result = open_app(app_name)
                    record_event("open_app", app_name, result)
                    robin_says(result)
                else:
                    record_event("open_app", app_name, "blocked")
                    robin_says("This action is not permitted.")
            except Exception as error:
                robin_says(f"The app action failed: {error}")

            continue

        messages.append({
            "role": "user",
            "content": command,
        })

        # Keep recent conversation context manageable.
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

        answer_parts = []
        generation_started = time.perf_counter()
        first_output_at = None

        thinking_displayed = 0
        thinking_limit = 300
        thinking_label_printed = False
        thinking_notice_shown = False
        answer_label_printed = False

        try:
            for kind, chunk in stream_reply(
                model,
                client,
                messages,
            ):
                if not chunk:
                    continue

                if kind == "thinking":
                    if not thinking_label_printed:
                        print(
                            f"{GRAY}Thinking > ",
                            end="",
                            flush=True,
                        )
                        thinking_label_printed = True

                    remaining = max(
                        0,
                        thinking_limit - thinking_displayed,
                    )
                    visible = chunk[:remaining]

                    if visible:
                        sys.stdout.write(f"{GRAY}{visible}")
                        sys.stdout.flush()
                        thinking_displayed += len(visible)

                    if (
                        len(visible) < len(chunk)
                        and not thinking_notice_shown
                    ):
                        sys.stdout.write(
                            f"{GRAY} ... [thinking display shortened]"
                        )
                        sys.stdout.flush()
                        thinking_notice_shown = True

                    continue

                # Start the final answer in green.
                if not answer_label_printed:
                    if thinking_label_printed:
                        print(RESET)

                    print(
                        f"{GREEN}Robin > ",
                        end="",
                        flush=True,
                    )
                    answer_label_printed = True

                if first_output_at is None:
                    first_output_at = time.perf_counter()

                answer_parts.append(chunk)
                sys.stdout.write(f"{GREEN}{chunk}")
                sys.stdout.flush()

            print(f"{RESET}\n")

            answer = "".join(answer_parts).strip()
            elapsed = time.perf_counter() - generation_started

            if first_output_at is not None:
                first_delay = first_output_at - generation_started
                print(
                    f"{YELLOW}[First answer text: {first_delay:.2f}s"
                    f" | Total: {elapsed:.2f}s]{RESET}\n"
                )

            if not answer:
                messages.pop()
                robin_says("I couldn't produce a final answer. Please try again.")
                continue

            messages.append({
                "role": "assistant",
                "content": answer,
            })

            try:
                save_turn(command, answer)
            except Exception as error:
                robin_says(f"Warning: I couldn't save conversation memory: {error}")

        except Exception as error:
            messages.pop()
            print(RESET)
            robin_says(f"I couldn't generate a response: {error}")


if __name__ == "__main__":
    main()
