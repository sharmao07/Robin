def list_installed_models(client):
    """List models already available in the local Ollama server."""

    response = client.list()
    names = []

    for item in response.models:
        name = (
            getattr(item, "model", None)
            or getattr(item, "name", None)
        )

        if name:
            names.append(name)

    return sorted(set(names))


def download_model(client, name, progress_callback=None):
    """Download a model only when explicitly requested."""

    for status in client.pull(name, stream=True):
        description = getattr(status, "status", "Downloading")
        completed = getattr(status, "completed", None)
        total = getattr(status, "total", None)

        progress = None
        if total:
            progress = int((completed or 0) * 100 / total)

        if progress_callback:
            progress_callback(description, progress)
