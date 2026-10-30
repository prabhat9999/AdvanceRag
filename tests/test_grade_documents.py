import importlib
from unittest.mock import Mock
from types import SimpleNamespace

from langchain_core.documents import Document

def test_empty_documents_request_web_search(monkeypatch):
    # The grader currently initializes ChatOpenAI during import.
    # Supply a placeholder so this test doesn't require a real API key.
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    module = importlib.import_module("graph.nodes.grade_documents")

    fake_grader = Mock(spec_set=["invoke"])
    monkeypatch.setattr(module, "retrieval_grader", fake_grader)

    state = {
        "question": "What is task decomposition?",
        "documents": [],
    }

    result = module.grade_documents(state)

    assert result["documents"] == []
    assert result["question"] == state["question"]
    assert result["web_search"] is True
    fake_grader.invoke.assert_not_called()

def test_mixed_documents_keep_only_relevant(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    module = importlib.import_module("graph.nodes.grade_documents")

    relevant = Document(page_content="Agents break tasks into smaller steps.")
    irrelevant = Document(page_content="Bananas are a type of fruit.")

    fake_grader = Mock(spec_set=["invoke"])
    fake_grader.invoke.side_effect = [
        SimpleNamespace(binary_score=True),
        SimpleNamespace(binary_score=False),
    ]

    monkeypatch.setattr(module, "retrieval_grader", fake_grader)

    result = module.grade_documents(
        {
            "question": "What is task decomposition?",
            "documents": [relevant, irrelevant],
        }
    )

    assert result["documents"] == [relevant]
    assert result["web_search"] is False
    assert fake_grader.invoke.call_count == 2

def test_all_irrelevant_documents_request_web_search(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    module = importlib.import_module("graph.nodes.grade_documents")

    documents = [
        Document(page_content="Bananas are a type of fruit."),
        Document(page_content="The ocean contains salt water."),
    ]

    fake_grader = Mock(spec_set=["invoke"])
    fake_grader.invoke.return_value = SimpleNamespace(binary_score=False)

    monkeypatch.setattr(module, "retrieval_grader", fake_grader)

    result = module.grade_documents(
        {
            "question": "What is task decomposition?",
            "documents": documents,
        }
    )

    assert result["documents"] == []
    assert result["web_search"] is True
    assert fake_grader.invoke.call_count == 2

def test_all_relevant_documents_are_kept(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    module = importlib.import_module("graph.nodes.grade_documents")

    documents = [
        Document(page_content="Task decomposition divides a task into steps."),
        Document(page_content="An agent can plan smaller subtasks."),
    ]

    fake_grader = Mock(spec_set=["invoke"])
    fake_grader.invoke.return_value = SimpleNamespace(binary_score=True)

    monkeypatch.setattr(module, "retrieval_grader", fake_grader)

    result = module.grade_documents(
        {
            "question": "What is task decomposition?",
            "documents": documents,
        }
    )

    assert result["documents"] == documents
    assert result["web_search"] is False
    assert fake_grader.invoke.call_count == 2