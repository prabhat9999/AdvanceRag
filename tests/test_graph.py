from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


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