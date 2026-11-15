from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSequence,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeHallucinations(BaseModel):
    """Whether an answer is supported by the supplied evidence."""

    binary_score: bool = Field(
        description=(
            "True if the generated answer is supported by "
            "the retrieved documents; otherwise False."
        )
    )


@lru_cache(maxsize=1)
def get_hallucination_grader() -> RunnableSequence:
    """Build the chain on first use and reuse it afterward."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    structured_llm_grader = llm.with_structured_output(
        GradeHallucinations
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Assess whether the generated answer is supported by "
                "the supplied facts. Return True when supported "
                "and False otherwise.",
            ),
            (
                "human",
                "Facts:\n{documents}\n\n"
                "Generated answer:\n{generation}",
            ),
        ]
    )

    return prompt | structured_llm_grader


def _invoke_hallucination_grader(
    inputs: dict,
    config: RunnableConfig,
):
    return get_hallucination_grader().invoke(inputs, config=config)


hallucination_grader = RunnableLambda(
    _invoke_hallucination_grader
)