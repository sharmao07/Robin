# Robin
**Local first AI assistant for macOS with offline intelligence, computer control, memory, and other permission based actions.**

Robin is an experimental desktop AI assistant focused on local AI, conversation memory, customizable personality, and permission-based computer actions.

The goal is to build an assistant that helps you get things done while keeping control over your data and the actions it can perform.

> **Status:** Experimental and under active development. Setup instructions and features may change as Robin evolves.

## Features

- **Local AI:** Connect to a language model running locally through Ollama.
- **Conversation memory:** Store conversation history locally.
- **Custom personality:** Configure Robin's behavior and preferences.
- **Permission-based actions:** Use a permission system to control desktop actions.
- **Activity auditing:** Keep a record of relevant actions for transparency and debugging.
- **Desktop interface:** Interact with Robin through a Python and PySide6 desktop application.

Capabilities depend on the current implementation and local configuration. This repository does not include model weights.

## Requirements

- macOS (primary target)
- Python 3.10 or later
- [Ollama](https://ollama.com/)
- A locally downloaded model supported by your Ollama installation

## Getting Started

### 1. Install Ollama

Download Ollama from [ollama.com](https://ollama.com/) and install it.

Download a model supported by your setup. For example:

```bash
ollama pull qwen3:4b
```

Check that the model is available:

```bash
ollama list
```

You can use another compatible model if preferred.

### 2. Clone Robin

```bash
git clone https://github.com/sharmao07/Robin.git
cd Robin
```

### 3. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Robin

```bash
python -m robin.desktop
```

Make sure Ollama is running and a local model is available if Robin cannot connect to the model provider.

## Development

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
python -m unittest discover tests/
```

Check Python files for syntax errors:

```bash
python -m compileall -q robin tests
```

Please run the tests in your own environment before submitting changes, and report any failures rather than assuming they are unrelated.

## Security and Privacy

Robin is designed around local execution and user control, but local AI is not automatically risk-free.

- Treat model-generated instructions and tool requests as untrusted.
- Review permission requests before approving actions.
- Never commit passwords, API keys, personal conversations, or private logs.
- Be cautious when allowing an AI assistant to control applications or access files.
- Report security vulnerabilities privately to the maintainers.

See [SECURITY.md](SECURITY.md) for the project's security policy.

## Roadmap

Robin is being developed toward a more capable personal AI assistant, with an emphasis on:

- Better long-term project and conversation continuity
- More reliable local inference
- Safer desktop automation and permissions
- A polished desktop experience
- Easier installation and distribution

These are development goals, not a claim that every capability is currently implemented or released.

## Contributing

Contributions, bug reports, and suggestions are welcome.

Before submitting a pull request:

1. Explain what you changed and why.
2. Run the relevant tests.
3. Include steps to reproduce fixes where applicable.
4. Keep credentials and private information out of commits.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Licence

Robin is licensed under the [MIT License](LICENSE). Third-party dependencies and models may have separate licence terms.
