from typing import cast

import pytest
from dotenv import load_dotenv

from ingestion import get_retriever
from graph.chains.retrieval_grader_chain import (
    GradeDocuments,
    retrieval_grader,
)
from graph.chains.generation_chain import generation_chain
from graph.chains.hallucination_grader_chain import (
    GradeHallucinations,
    hallucination_grader,
)
from graph.chains.router_chain import RouteQuery, question_router
pytestmark = pytest.mark.integration
load_dotenv()


@pytest.fixture
def retriever():
    """Create the retriever when a test requests it."""
    return get_retriever()


def test_retrieval_grader_yes(retriever) -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    assert docs, "Expected documents; ingest the source articles first."

    result = cast(
        GradeDocuments,
        retrieval_grader.invoke(
            {
                "question": question,
                "document": docs[0].page_content,
            }
        ),
    )

    assert result.binary_score is True


def test_retrieval_grader_no() -> None:
    # Use a known unrelated passage to isolate the relevance check.
    result = cast(
        GradeDocuments,
        retrieval_grader.invoke(
            {
                "question": "How do I prepare pizza dough?",
                "document": (
                    "Agent memory stores information from previous "
                    "interactions to support future decisions."
                ),
            }
        ),
    )

    assert result.binary_score is False


def test_generation_chain(retriever) -> None:
    question = "What is agent memory?"
    docs = retriever.invoke(question)

    assert docs, "Expected documents; ingest the source articles first."

    generation = generation_chain.invoke(
        {"question": question, "context": docs}
    )

    assert isinstance(generation, str)
    assert generation.strip(), "Expected a nonempty answer."


def test_hallucination_grader_answer_yes() -> None:
    # Supply a directly supported answer.
    result = cast(
        GradeHallucinations,
        hallucination_grader.invoke(
            {
                "documents": (
                    "Agent memory stores information from "
                    "previous interactions."
                ),
                "generation": (
                    "Agent memory stores information from "
                    "previous interactions."
                ),
            }
        ),
    )

    assert result.binary_score is True


def test_hallucination_grader_answer_no() -> None:
    # Supply an explicitly unsupported answer.
    result = cast(
        GradeHallucinations,
        hallucination_grader.invoke(
            {
                "documents": (
                    "Agent memory stores information from "
                    "previous interactions."
                ),
                "generation": (
                    "Agent memory is a pizza recipe containing "
                    "tomatoes and mozzarella."
                ),
            }
        ),
    )

    assert result.binary_score is False


def test_router_to_vectorstore() -> None:
    result = cast(
        RouteQuery,
        question_router.invoke({"question": "What is agent memory?"}),
    )

    assert result.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    result = cast(
        RouteQuery,
        question_router.invoke({"question": "How do I make pizza?"}),
    )

    assert result.datasource == "websearch"