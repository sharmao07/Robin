from robin.tools.apps import open_app


def main():
    print("=" * 36)
    print("ROBIN - YOUR LOCAL AI ASSISTANT")
    print("=" * 36)
    print("Current mode: Basic local prototype")
    print("Type 'help' for commands.")
    print("Type 'exit' to quit.\n")

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

        elif lowered == "help":
            print("Robin > Available commands:")
            print("  open vscode")
            print("  open safari")
            print("  open notes")
            print("  open finder")
            print("  open calculator")
            print("  open blender")
            print("  help")
            print("  exit")

        elif lowered.startswith("open "):
            app_name = command[5:].strip()

            if app_name:
                print(f"Robin > {open_app(app_name)}")
            else:
                print("Robin > Tell me which approved app to open.")

        else:
            print(
                "Robin > My local language model is not connected yet. "
                "For now, I can open approved apps. Type 'help'."
            )


if __name__ == "__main__":
    main()
