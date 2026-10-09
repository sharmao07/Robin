import subprocess


# Robin can only open apps listed here.
ALLOWED_APPS = {
    "vscode": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
    "safari": "Safari",
    "notes": "Notes",
    "finder": "Finder",
    "calculator": "Calculator",
    "blender": "Blender",
}


def open_app(name: str) -> str:
    alias = name.strip().lower()

    if alias not in ALLOWED_APPS:
        allowed = ", ".join(sorted(ALLOWED_APPS))
        return (
            f"App '{name}' is not on Robin's approved list. "
            f"Approved names: {allowed}"
        )

    app_name = ALLOWED_APPS[alias]

    try:
        subprocess.run(
            ["/usr/bin/open", "-a", app_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
        return f"Opening {app_name}."

    except subprocess.CalledProcessError:
        return (
            f"Could not open {app_name}. "
            "Check whether it is installed on your Mac."
        )

    except subprocess.TimeoutExpired:
        return f"Opening {app_name} timed out."
