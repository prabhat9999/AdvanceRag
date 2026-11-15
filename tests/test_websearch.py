import importlib
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document


def test_web_search_preserves_input_documents(monkeypatch):
    module = importlib.import_module("graph.nodes.websearch")

    original_document = Document(page_content="Existing evidence.")
    original_documents = [original_document]

    fake_tool = Mock()
    fake_tool.invoke.return_value = {
        "results": [
            {"content": "First web passage."},
            {"content": "Second web passage."},
        ]
    }

    monkeypatch.setattr(
        module, "get_web_search_tool", lambda: fake_tool
    )

    result = module.web_search(
        {
            "question": "What is agent memory?",
            "documents": original_documents,
        }
    )

    fake_tool.invoke.assert_called_once_with(
        {"query": "What is agent memory?"}
    )

    assert original_documents == [original_document]
    assert result["documents"] is not original_documents
    assert len(result["documents"]) == 2
    assert result["documents"][0] is original_document
    assert result["documents"][1].page_content == (
        "First web passage.\nSecond web passage."
    )
    assert result["question"] == "What is agent memory?"


def test_empty_search_results_add_no_document(monkeypatch):
    module = importlib.import_module("graph.nodes.websearch")

    fake_tool = Mock()
    fake_tool.invoke.return_value = {"results": []}

    monkeypatch.setattr(
        module, "get_web_search_tool", lambda: fake_tool
    )

    result = module.web_search(
        {"question": "A question with no search results"}
    )

    assert result["documents"] == []


def test_web_search_client_is_created_once(monkeypatch):
    module = importlib.import_module("graph.nodes.websearch")
    module.get_web_search_tool.cache_clear()

    fake_client = Mock()
    fake_constructor = Mock(return_value=fake_client)
    monkeypatch.setattr(module, "TavilySearch", fake_constructor)

    try:
        first = module.get_web_search_tool()
        second = module.get_web_search_tool()

        fake_constructor.assert_called_once_with(
            max_results=3,
            search_type="web",
            search_engine="google",
        )
        assert first is fake_client
        assert second is first
    finally:
        module.get_web_search_tool.cache_clear()

def test_empty_search_preserves_existing_evidence(monkeypatch):
    from graph.routing import decide_after_web_search

    module = importlib.import_module("graph.nodes.websearch")

    document = Document(page_content="Existing evidence.")
    original_documents = [document]

    fake_tool = Mock()
    fake_tool.invoke.return_value = {"results": []}

    monkeypatch.setattr(
        module, "get_web_search_tool", lambda: fake_tool
    )

    state = {
        "question": "What is agent memory?",
        "documents": original_documents,
    }

    update = module.web_search(state)
    next_step = decide_after_web_search({**state, **update})

    assert update["question"] == state["question"]
    assert update["documents"] == [document]
    assert update["documents"][0] is document
    assert update["documents"] is not original_documents
    assert original_documents == [document]
    assert next_step == "generate"

    fake_tool.invoke.assert_called_once_with(
        {"query": state["question"]}
    )

def test_search_error_is_reported_clearly(monkeypatch):
    from requests.exceptions import Timeout

    module = importlib.import_module("graph.nodes.websearch")

    fake_tool = Mock()
    fake_tool.invoke.return_value = {
        "error": Timeout("Search request timed out.")
    }

    monkeypatch.setattr(
        module,
        "get_web_search_tool",
        lambda: fake_tool,
    )

    document = Document(page_content="Existing evidence.")
    state = {
        "question": "What is agent memory?",
        "documents": [document],
    }

    with pytest.raises(
        RuntimeError,
        match=r"^Web search is unavailable\. Please try again\.$",
    ):
        module.web_search(state)

    fake_tool.invoke.assert_called_once_with(
        {"query": state["question"]}
    )
    assert state["documents"] == [document]