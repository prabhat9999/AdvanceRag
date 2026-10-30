from graph.consts import GENERATE, WEBSEARCH
from graph.routing import decide_to_generate
import pytest
from graph.routing import decide_after_rejection

def test_web_search_flag_routes_to_web_search():
    state = {
        "question": "What is task decomposition?",
        "web_search": True,
    }

    assert decide_to_generate(state) == WEBSEARCH


def test_no_web_search_flag_routes_to_generation():
    state = {
        "question": "What is task decomposition?",
        "web_search": False,
    }

    assert decide_to_generate(state) == GENERATE

def test_missing_web_search_flag_raises_error():
    state = {
        "question": "What is task decomposition?",
    }

    with pytest.raises(
        ValueError,
        match="Document grading must set web_search before routing.",
    ):
        decide_to_generate(state)

@pytest.mark.parametrize(
    "attempts, expected_route",
    [
        (1, "retrieve"),
        (2, "insufficient_evidence"),
        (3, "insufficient_evidence"),
    ],
)
def test_rejected_answer_respects_attempt_limit(
    attempts,
    expected_route,
):
    state = {
        "question": "What is task decomposition?",
        "generation_attempts": attempts,
    }

    assert decide_after_rejection(state) == expected_route