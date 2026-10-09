from robin.brain.local_llm import load_brain, generate_reply
from robin.tools.apps import open_app


SYSTEM_PROMPT = """
You are Robin, a personal AI assistant running locally on a Mac.

Personality:
- Speak naturally, calmly, intelligently, and confidently.
- Be helpful like a capable personal desktop assistant.
- Explain technical subjects clearly.
- Assist with programming, planning, writing, and research.
- Be honest about uncertainty and limitations.

Operating rules:
- You are currently running without an online search tool.
- Never pretend you accessed information you did not access.
- Never claim you performed an action unless it actually happened.
- You cannot automatically execute commands from your replies.
- The only computer tool currently available is explicit app opening.
- Do not claim to have read, edited, or deleted any files.
- Destructive actions must remain unavailable by default.
"""


def main():
    print("=" * 42)
    print("ROBIN | LOCAL PERSONAL AI")
    print("=" * 42)
    print("Loading Robin's local brain...")
    print("The first model load may take some time.\n")

    try:
        model, tokenizer = load_brain()
    except Exception as error:
        print(f"Could not load Robin's brain: {error}")
        return

    print("Robin > I'm online locally. How can I help?")
    print("Type 'help' for commands or 'exit' to quit.\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            command = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRobin > Goodbye.")
            break

        if not command:
            continue

        lowered = command.lower()

        if lowered in ("exit", "quit"):
            print("Robin > Goodbye.")
            break

        if lowered == "help":
            print("Robin > You can chat with me or use:")
            print("  open safari")
            print("  open vscode")
            print("  open notes")
            print("  open finder")
            print("  open calculator")
            print("  open blender")
            print("  help")
            print("  exit")
            continue

        if lowered.startswith("open "):
            result = open_app(command[5:].strip())
            print(f"Robin > {result}\n")
            continue

        messages.append({
            "role": "user",
            "content": command,
        })

        # Keep recent conversation context manageable.
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

        try:
            answer = generate_reply(
                model,
                tokenizer,
                messages,
            )
        except Exception as error:
            messages.pop()
            print(f"Robin > I couldn't generate a reply: {error}\n")
            continue

        messages.append({
            "role": "assistant",
            "content": answer,
        })

        print(f"Robin > {answer}\n")


if __name__ == "__main__":
    main()
