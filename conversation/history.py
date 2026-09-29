import json
from pathlib import Path


DEFAULT_HISTORY_FILE = Path("conversation_history.json")


def load_history(
    history_file: Path = DEFAULT_HISTORY_FILE,
) -> list[dict[str, str]]:
    """Load conversation history from a JSON file."""

    if not history_file.exists():
        return []

    try:
        with history_file.open("r", encoding="utf-8") as file:
            history = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(history, list):
        return []

    return history


def save_history(
    history: list[dict[str, str]],
    history_file: Path = DEFAULT_HISTORY_FILE,
) -> None:
    """Save conversation history to a JSON file."""

    with history_file.open("w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            indent=2,
            ensure_ascii=False,
        )


def add_message(
    history: list[dict[str, str]],
    role: str,
    content: str,
) -> list[dict[str, str]]:
    """Add one message to the conversation history."""

    history.append(
        {
            "role": role,
            "content": content,
        }
    )

    return history


def format_history(
    history: list[dict[str, str]],
    max_messages: int = 6,
) -> str:
    """Format recent conversation history for the LLM."""

    if not history:
        return "No previous conversation."

    recent_messages = history[-max_messages:]

    lines = []

    for message in recent_messages:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")

        lines.append(f"{role}: {content}")

    return "\n".join(lines)