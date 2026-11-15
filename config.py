import os


def validate_environment(required_keys: tuple[str, ...]) -> None:
    """Reject missing, empty, or whitespace-only settings."""
    missing = []

    for key in required_keys:
        value = os.getenv(key)

        if value is None or not value.strip():
            missing.append(key)

    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Missing required settings: {names}. "
            "Set them in your environment or local .env file."
        )