import ollama


MODEL_NAME = "qwen3:4b"
client = ollama.Client(host="http://127.0.0.1:11434")


def load_brain():
    """Verify that Robin's model is available locally."""

    client.show(MODEL_NAME)
    return MODEL_NAME, client


def stream_reply(model, client, messages):
    """Stream responses from Robin's local model."""

    response_stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        think=False,
    )

    for part in response_stream:
        content = part.message.content or ""

        if content:
            yield content
