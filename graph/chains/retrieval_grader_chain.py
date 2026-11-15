from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSequence,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Whether a retrieved document is relevant to the question."""

    binary_score: bool = Field(
        description=(
            "True if the document is relevant to the question; "
            "otherwise False."
        )
    )


@lru_cache(maxsize=1)
def get_retrieval_grader() -> RunnableSequence:
    """Build the retrieval grader on first use and reuse it."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        timeout=30.0,
        max_retries=1,
    )

    structured_llm_grader = llm.with_structured_output(
        GradeDocuments
    )

    system = (
        "You are a grader assessing relevance of a retrieved document "
        "to a user question. "
        "If the document contains keywords or semantic meaning related "
        "to the question, grade it as relevant. "
        "Give a binary score True or False to indicate whether the "
        "document is relevant to the question."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document:\n{document}\n\n"
                "User question:\n{question}",
            ),
        ]
    )

    return prompt | structured_llm_grader


def _invoke_retrieval_grader(
    inputs: dict,
    config: RunnableConfig,
) -> GradeDocuments:
    return get_retrieval_grader().invoke(inputs, config=config)


retrieval_grader = RunnableLambda(_invoke_retrieval_grader)