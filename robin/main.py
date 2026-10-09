import sys

from robin.brain.local_llm import load_brain, generate_reply
from robin.tools.apps import open_app


# ANSI terminal colours
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"


# Disable ANSI colours when output is redirected.
if not sys.stdout.isatty():
    RESET = CYAN = GREEN = MAGENTA = YELLOW = ""


SYSTEM_PROMPT = """
You are Robin, a personal AI assistant running locally on a Mac.

Personality:
- Speak naturally, intelligently, and confidently.
- Be conversational like a capable personal desktop assistant.
- Explain coding and technical subjects clearly.
- Help with writing, planning, programming, and research.
- Be honest about uncertainty.

Response formatting:
- Return only your user-facing answer.
- Never display <think> or </think> tags.
- Do not print internal planning or analysis tags.
- Keep answers useful and easy to read.

Operating rules:
- You currently have no online search tool.
- Never pretend to have accessed information you did not access.
- Never claim an action succeeded unless it actually did.
- Your only computer tool is explicitly requested app opening.
- You cannot read, edit, or delete arbitrary files.
- Destructive actions must remain unavailable by default.
"""


def robin_says(message):
    """Print Robin's response in green."""
    print(f"{GREEN}Robin > {message}{RESET}\n")


def main():
    print(f"{MAGENTA}")
    print("==========================================")
    print("       R O B I N  |  LOCAL AI")
    print("==========================================")
    print(f"{RESET}")

    print(f"{YELLOW}Loading Robin's local brain...{RESET}")
    print(f"{YELLOW}Please wait while the model loads.{RESET}\n")

    try:
        model, tokenizer = load_brain()
    except Exception as error:
        print(f"{YELLOW}Could not load Robin's brain: {error}{RESET}")
        return

    robin_says("I'm ready. How can I help?")

    print(f"{YELLOW}Type 'help' for commands or 'exit' to quit.{RESET}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            # Cyan colours the prompt AND the text you type.
            print(f"{CYAN}You > ", end="", flush=True)
            command = input().strip()
            print(RESET, end="", flush=True)

        except (EOFError, KeyboardInterrupt):
            print(f"\n{RESET}")
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
                "Available commands:\n"
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

        # Keep explicit app opening separate from the language model.
        if lowered.startswith("open "):
            app_name = command[5:].strip()
            result = open_app(app_name)
            robin_says(result)
            continue

        messages.append({
            "role": "user",
            "content": command,
        })

        # Keep the recent conversation within a manageable size.
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

        try:
            answer = generate_reply(
                model,
                tokenizer,
                messages,
            )

            if not answer:
                messages.pop()
                robin_says("I couldn't produce a clean response. Please try again.")
                continue

        except Exception as error:
            messages.pop()
            robin_says(f"I couldn't generate a response: {error}")
            continue

        messages.append({
            "role": "assistant",
            "content": answer,
        })

        robin_says(answer)


if __name__ == "__main__":
    main()
