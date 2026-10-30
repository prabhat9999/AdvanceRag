from typing import Literal

from graph.state import GraphState


MAX_GENERATION_ATTEMPTS = 2


def decide_to_generate(
    state: GraphState,
) -> Literal["websearch", "generate"]:
    """Choose the next step after document grading."""

    if "web_search" not in state:
        raise ValueError(
            "Document grading must set web_search before routing."
        )

    if state["web_search"]:
        return "websearch"

    return "generate"


def decide_after_rejection(
    state: GraphState,
) -> Literal["retrieve", "insufficient_evidence"]:
    """Retry a rejected answer only while attempts remain."""

    if "generation_attempts" not in state:
        raise ValueError(
            "Generation attempts must be tracked before retry routing."
        )

    if state["generation_attempts"] < MAX_GENERATION_ATTEMPTS:
        return "retrieve"

    return "insufficient_evidence"