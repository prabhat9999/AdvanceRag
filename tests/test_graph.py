import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document


def test_unsupported_answer():
    from graph.graph import (
        grade_generation_grounded_in_documents_and_question,
    )

    state = {
        "question": "What is agent memory?",
        "documents": [
            Document(page_content="Agent memory stores information.")
        ],
        "generation": "Agent memory is a type of pizza.",
    }

    with patch(
        "graph.graph.hallucination_grader"
    ) as mock_hallucination:
        with patch("graph.graph.answer_grader") as mock_answer:
            mock_hallucination.invoke.return_value.binary_score = False

            result = grade_generation_grounded_in_documents_and_question(
                state
            )

            assert result == "not supported"

            # An unsupported answer should not reach usefulness grading.
            mock_answer.invoke.assert_not_called()


def test_insufficient_evidence_replaces_rejected_answer():
    from graph.graph import insufficient_evidence

    state = {
        "question": "What is agent memory?",
        "generation": "An unsupported answer.",
        "generation_attempts": 2,
    }

    result = insufficient_evidence(state)

    assert (
        "couldn't produce a supported answer"
        in result["generation"]
    )

    # The node returns an update rather than changing its input directly.
    assert state["generation"] == "An unsupported answer."


@pytest.mark.parametrize(
    "grade, attempts, expected",
    [
        ("useful", 1, "useful"),
        ("useful", 2, "useful"),
        ("not supported", 1, "retrieve"),
        ("not supported", 2, "insufficient_evidence"),
        ("not useful", 1, "retrieve"),
        ("not useful", 2, "insufficient_evidence"),
    ],
)
def test_generation_routing(monkeypatch, grade, attempts, expected):
    graph_module = importlib.import_module("graph.graph")

    # Control the grading result so this test isolates routing behavior.
    monkeypatch.setattr(
        graph_module,
        "grade_generation_grounded_in_documents_and_question",
        lambda state: grade,
    )

    result = graph_module.decide_after_generation(
        {
            "question": "What is agent memory?",
            "generation_attempts": attempts,
        }
    )

    assert result == expected


@pytest.mark.parametrize(
    "grounding_results, expected_answer",
    [
        ([True], "A generated answer."),
        ([False, False], "couldn't produce a supported answer"),
        ([False, True], "A generated answer."),
    ],
)
def test_graph_respects_attempt_limit(
    monkeypatch,
    grounding_results,
    expected_answer,
):
    graph_module = importlib.import_module("graph.graph")
    generation_module = importlib.import_module(
        "graph.nodes.generate"
    )

    document = Document(
        page_content="Agent memory stores information."
    )

    # Always route to document retrieval.
    fake_router = Mock()
    fake_router.invoke.return_value = SimpleNamespace(
        datasource="vectorstore"
    )
    monkeypatch.setattr(
        graph_module,
        "question_router",
        fake_router,
    )

    # Supply evidence without accessing the real vector database.
    monkeypatch.setattr(
        graph_module,
        "retrieve",
        lambda state: {"documents": [document]},
    )

    # Treat the retrieved document as relevant.
    monkeypatch.setattr(
        graph_module,
        "grade_documents",
        lambda state: {"web_search": False},
    )

    # Keep the real generation node and its attempt counter.
    # Replace only the chain that would call the external LLM.
    fake_generation = Mock()
    fake_generation.invoke.return_value = "A generated answer."
    monkeypatch.setattr(
        generation_module,
        "generation_chain",
        fake_generation,
    )

    # Each invocation receives the next configured grounding decision.
    fake_grounding_grader = Mock()
    fake_grounding_grader.invoke.side_effect = [
        SimpleNamespace(binary_score=value)
        for value in grounding_results
    ]
    monkeypatch.setattr(
        graph_module,
        "hallucination_grader",
        fake_grounding_grader,
    )

    # Any answer that passes grounding also passes usefulness grading.
    fake_answer_grader = Mock()
    fake_answer_grader.invoke.return_value = SimpleNamespace(
        binary_score=True
    )
    monkeypatch.setattr(
        graph_module,
        "answer_grader",
        fake_answer_grader,
    )

    # Build after replacing dependencies so the graph uses our fakes.
    app = graph_module.build_graph()

    result = app.invoke(
        {"question": "What is agent memory?"},
        config={"recursion_limit": 15},
    )

    expected_attempts = len(grounding_results)

    # Check actual execution and the graph's recorded attempt count.
    assert fake_generation.invoke.call_count == expected_attempts
    assert result["generation_attempts"] == expected_attempts

    assert expected_answer in result["generation"]

    # Usefulness grading runs only after grounding succeeds.
    assert fake_answer_grader.invoke.call_count == sum(
        grounding_results
    )