from functools import lru_cache
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSequence,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        description=(
            "Choose vectorstore or websearch based on the question."
        )
    )


@lru_cache(maxsize=1)
def get_question_router() -> RunnableSequence:
    """Build the question router on first use and reuse it."""
    llm = ChatOpenAI(
   model="gpt-4o-mini",
        temperature=0,
        timeout=30.0,
        max_retries=1,
    )

    structured_llm_router = llm.with_structured_output(RouteQuery)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at routing a user question to a "
                "vectorstore or web search. "
                "The vectorstore contains documents related to agents, "
                "prompt engineering, and adversarial attacks. "
                "Use the vectorstore for questions on these topics. "
                "For all else, use web search.",
            ),
            ("human", "{question}"),
        ]
    )

    return prompt | structured_llm_router


def _invoke_question_router(
    inputs: dict,
    config: RunnableConfig,
) -> RouteQuery:
    return get_question_router().invoke(inputs, config=config)


question_router = RunnableLambda(_invoke_question_router)