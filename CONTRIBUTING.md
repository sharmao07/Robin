# Contributing to Robin

Thanks for helping improve Robin.

## Development setup

1. Install a compatible Python version and Ollama.
2. Clone this repository.
3. Create and activate a virtual environment.
4. Install `requirements.txt`.
5. Install and test a local language model.
6. Run `python -m robin.desktop`.

## Before submitting changes

- Run Python compilation checks.
- Run relevant unit tests.
- Describe the change and how it was tested.
- Do not commit model weights, personal conversations, API keys,
  logs, or private files.

## Security

All desktop automation must use Robin's permission system.
Do not add tools that can silently delete files or bypass approval.
