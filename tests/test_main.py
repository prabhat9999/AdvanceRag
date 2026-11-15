import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

import main as main_module


def test_accepts_question():
    args = main_module.parse_args(["What is agent memory?"])

    assert args.question == "What is agent memory?"


def test_requires_question():
    with pytest.raises(SystemExit) as error:
        main_module.parse_args([])

    assert error.value.code == 2


@pytest.mark.parametrize("question", ["", "   "])
def test_rejects_blank_question(question):
    with pytest.raises(SystemExit) as error:
        main_module.parse_args([question])

    assert error.value.code == 2


@pytest.fixture
def fake_graph(monkeypatch):
    """Replace the graph module so tests cannot call real services."""
    app = Mock()
    app.invoke.return_value = {
        "generation": "Agent memory stores information."
    }

    graph_module = ModuleType("graph.graph")
    graph_module.app = app

    monkeypatch.setitem(
        sys.modules,
        "graph.graph",
        graph_module,
    )

    return app


def test_missing_configuration_stops_graph(
    monkeypatch,
    fake_graph,
    capsys,
):
    # Prevent the developer's .env file from influencing this test.
    fake_load_dotenv = Mock()
    monkeypatch.setattr(
        main_module,
        "load_dotenv",
        fake_load_dotenv,
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(SystemExit) as error:
        main_module.main(["What is agent memory?"])

    message = str(error.value)

    assert "Configuration error:" in message
    assert "OPENAI_API_KEY" in message
    assert "TAVILY_API_KEY" in message

    fake_load_dotenv.assert_called_once_with()
    fake_graph.invoke.assert_not_called()

    assert "Final Output:" not in capsys.readouterr().out


def test_valid_configuration_runs_graph(
    monkeypatch,
    fake_graph,
    capsys,
):
    # Start without keys. Simulate loading them from .env.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def load_test_settings():
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    fake_load_dotenv = Mock(side_effect=load_test_settings)
    monkeypatch.setattr(
        main_module,
        "load_dotenv",
        fake_load_dotenv,
    )

    main_module.main(["  What is agent memory?  "])

    fake_load_dotenv.assert_called_once_with()

    fake_graph.invoke.assert_called_once_with(
        input={"question": "What is agent memory?"}
    )

    output = capsys.readouterr().out
    assert (
        "Final Output: Agent memory stores information."
        in output
    )