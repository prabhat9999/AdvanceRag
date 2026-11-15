from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSequence,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):
    """Whether the generated answer addresses the question."""

    binary_score: bool = Field(
        description=(
            "True if the answer addresses the question; otherwise False."
        )
    )


@lru_cache(maxsize=1)
def get_answer_grader() -> RunnableSequence:
    """Build the answer grader on first use and reuse it."""
    llm = ChatOpenAI(
       model="gpt-4o-mini",
        temperature=0,
        timeout=30.0,
        max_retries=1,
    )

    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Assess whether the generated answer addresses the "
                "user's question. Return True if it addresses the "
                "question and False otherwise. Evidence grounding "
                "is checked separately.",
            ),
            (
                "human",
                "User question:\n{question}\n\n"
                "Generated answer:\n{generation}",
            ),
        ]
    )

    return prompt | structured_llm_grader


def _invoke_answer_grader(
    inputs: dict,
    config: RunnableConfig,
) -> GradeAnswer:
    return get_answer_grader().invoke(inputs, config=config)


answer_grader = RunnableLambda(_invoke_answer_grader)