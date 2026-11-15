from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
import pytest

def test_unsupported_answer():
    # Import here for now because graph.py has import-time side effects.
    from graph.graph import grade_generation_grounded_in_documents_and_question

    state = {
        "question": "What is agent memory?",
        "documents": [
            Document(page_content="Agent memory stores information.")
        ],
        "generation": "Agent memory is a type of pizza."
    }

    with patch("graph.graph.hallucination_grader") as mock_hallucination:
        with patch("graph.graph.answer_grader") as mock_answer:

            mock_hallucination.invoke.return_value.binary_score = False

            result = grade_generation_grounded_in_documents_and_question(state)

            assert result == "not supported"
            mock_answer.invoke.assert_not_called()

def test_insufficient_evidence_replaces_rejected_answer():
    from graph.graph import insufficient_evidence

    state = {
        "question": "What is agent memory?",
        "generation": "An unsupported answer.",
        "generation_attempts": 2,
     }

    result = insufficient_evidence(state)

    assert "couldn't produce a supported answer" in result["generation"]
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
    import importlib

    graph_module = importlib.import_module("graph.graph")

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
        ([False, False], "couldn't produce a supported answer"),
        ([False, True], "A generated answer."),
    ],
)
def test_graph_respects_attempt_limit(
    monkeypatch, grounding_results, expected_answer
):
    import importlib
    from types import SimpleNamespace
    from unittest.mock import Mock

    from langchain_core.documents import Document

    graph_module = importlib.import_module("graph.graph")
    generation_module = importlib.import_module("graph.nodes.generate")

    document = Document(
        page_content="Agent memory stores information."
    )

    # Always route the question to document retrieval.
    fake_router = Mock()
    fake_router.invoke.return_value = SimpleNamespace(
        datasource="vectorstore"
    )
    monkeypatch.setattr(
        graph_module, "question_router", fake_router
    )

    # Supply a document without accessing the real vector database.
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

    # Keep the real generation node and its attempt counter,
    # but replace its LLM chain.
    fake_generation = Mock()
    fake_generation.invoke.return_value = "A generated answer."
    monkeypatch.setattr(
        generation_module, "generation_chain", fake_generation
    )

    # Return one grounding decision per generation attempt.
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

    # If grounding passes, treat the answer as useful.
    fake_answer_grader = Mock()
    fake_answer_grader.invoke.return_value = SimpleNamespace(
        binary_score=True
    )
    monkeypatch.setattr(
        graph_module, "answer_grader", fake_answer_grader
    )

    # Build after replacing dependencies so the graph uses our fakes.
    app = graph_module.build_graph()

    result = app.invoke(
        {"question": "What is agent memory?"},
        config={"recursion_limit": 15},
    )

    assert fake_generation.invoke.call_count == 2
    assert result["generation_attempts"] == 2
    assert expected_answer in result["generation"]

    # The usefulness grader should run only when grounding passes.
    assert fake_answer_grader.invoke.call_count == sum(
        grounding_results
    )
    import importlib
    from types import SimpleNamespace
    from unittest.mock import Mock

    from langchain_core.documents import Document

    graph_module = importlib.import_module("graph.graph")
    generation_module = importlib.import_module("graph.nodes.generate")

    document = Document(
        page_content="Agent memory stores information."
    )

    # Arrange: always choose document retrieval.
    fake_router = Mock()
    fake_router.invoke.return_value = SimpleNamespace(
        datasource="vectorstore"
    )
    monkeypatch.setattr(
        graph_module, "question_router", fake_router
    )

    # Arrange: supply evidence without accessing a database or an LLM.
    monkeypatch.setattr(
        graph_module,
        "retrieve",
        lambda state: {"documents": [document]},
    )
    monkeypatch.setattr(
        graph_module,
        "grade_documents",
        lambda state: {"web_search": False},
    )

    # Keep the real generate node, but replace its external LLM chain.
    fake_generation = Mock()
    fake_generation.invoke.return_value = "An unsupported answer."
    monkeypatch.setattr(
        generation_module, "generation_chain", fake_generation
    )

    # Arrange: reject every generated answer as unsupported.
    fake_grounding_grader = Mock()
    fake_grounding_grader.invoke.return_value = SimpleNamespace(
        binary_score=False
    )
    monkeypatch.setattr(
        graph_module,
        "hallucination_grader",
        fake_grounding_grader,
    )

    # Act: build and execute the graph with these fake dependencies.
    app = graph_module.build_graph()
    result = app.invoke(
        {"question": "What is agent memory?"},
        config={"recursion_limit": 15},
    )

    # Assert: two attempts, followed by a clear fallback response.
    assert fake_generation.invoke.call_count == 2
    assert result["generation_attempts"] == 2
    assert "couldn't produce a supported answer" in result["generation"]