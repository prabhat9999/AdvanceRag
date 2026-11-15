import pytest

from config import validate_environment


@pytest.mark.parametrize("value", [None, "", "   "])
def test_rejects_missing_or_blank_setting(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", value)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        validate_environment(("OPENAI_API_KEY",))


def test_reports_all_missing_settings(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError) as error:
        validate_environment(
            ("OPENAI_API_KEY", "TAVILY_API_KEY")
        )

    message = str(error.value)
    assert "OPENAI_API_KEY" in message
    assert "TAVILY_API_KEY" in message


def test_accepts_present_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    validate_environment(
        ("OPENAI_API_KEY", "TAVILY_API_KEY")
    )