from collections.abc import Callable


POLICY = {
    "open_app": "allow",
    "read_file": "confirm",
    "write_file": "confirm",
    "run_command": "confirm",
    "install_package": "confirm",
    "send_message": "confirm",
    "delete_file": "deny",
    "permanent_delete": "deny",
}


def authorize(
    action: str,
    target: str = "",
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Apply a policy independent of the language model."""

    decision = POLICY.get(action, "deny")

    if decision == "allow":
        return True

    if decision == "deny":
        return False

    response = input_fn(
        f"\nRobin requests permission to perform '{action}' "
        f"on '{target}'.\n"
        "Type YES to approve this action: "
    )

    return response.strip() == "YES"
