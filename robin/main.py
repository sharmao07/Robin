import sys

from robin.brain.local_llm import load_brain, stream_reply
from robin.tools.apps import open_app
from robin.brain.memory import (
    initialize_memory,
    load_recent_messages,
    save_turn,
)


RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"

if not sys.stdout.isatty():
    RESET = CYAN = GREEN = MAGENTA = YELLOW = ""


SYSTEM_PROMPT = """
You are Robin, a personal AI assistant running locally on a Mac.

Speak naturally, intelligently, and confidently.
Help with coding, writing, planning, and research.
Explain technical subjects clearly.
Be honest about uncertainty and limitations.
Never pretend to have performed an action.
Never claim to know information you have not accessed.

Your available computer tool is restricted to opening
applications explicitly requested by the user.
You cannot execute arbitrary commands or delete files.
"""


def robin_says(message):
    print(f"{GREEN}Robin > {message}{RESET}\n")


def main():
    print(f"{MAGENTA}")
    print("==========================================")
    print("          ROBIN | LOCAL AI")
    print("==========================================")
    print(f"{RESET}")

    print(f"{YELLOW}Loading Robin's local brain...{RESET}")

    try:
        model, tokenizer = load_brain()
    except Exception as error:
        print(f"{YELLOW}Could not load Robin's brain: {error}{RESET}")
        return

    robin_says("I'm ready. How can I help?")
    print(f"{YELLOW}Type 'help' for commands or 'exit' to quit.{RESET}\n")

    initialize_memory()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(load_recent_messages(limit=8))

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
                "Commands:\n"
                "  open safari\n"
                "  open vscode\n"
                "  open notes\n"
                "  open finder\n"
                "  open calculator\n"
                "  open blender\n"
                "  help\n"
                "  exit"
            )
            continue

        if lowered.startswith("open "):
            result = open_app(command[5:].strip())
            robin_says(result)
            continue

        messages.append({"role": "user", "content": command})

        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

        answer_parts = []

        try:
            print(f"{GREEN}Robin > ", end="", flush=True)

            for chunk in stream_reply(model, tokenizer, messages):
                answer_parts.append(chunk)
                sys.stdout.write(chunk)
                sys.stdout.flush()

            answer = "".join(answer_parts).strip()
            print(f"{RESET}\n")

            if not answer:
                messages.pop()
                robin_says("I couldn't produce a response. Please try again.")
                continue

            messages.append({"role": "assistant", "content": answer})
            save_turn(command, answer)

        except Exception as error:
            messages.pop()
            print(RESET)
            robin_says(f"I couldn't generate a response: {error}")


if __name__ == "__main__":
    main()
