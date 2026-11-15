import importlib
from unittest.mock import Mock

from graph.routing import decide_to_generate


def test_empty_retrieval_routes_to_web_search(monkeypatch):
    grading_module = importlib.import_module(
        "graph.nodes.grade_documents"
    )

    fake_grader = Mock()
    monkeypatch.setattr(
        grading_module,
        "retrieval_grader",
        fake_grader,
    )

    state = {
        "question": "What is agent memory?",
        "documents": [],
    }

    # Act: grade the empty retrieval result, then choose the next step.
    update = grading_module.grade_documents(state)
    next_step = decide_to_generate({**state, **update})

    # Assert: preserve the question and request web search.
    assert update["question"] == state["question"]
    assert update["documents"] == []
    assert update["web_search"] is True
    assert next_step == "websearch"

    # With no documents, there is nothing to send to the grader.
    fake_grader.invoke.assert_not_called()