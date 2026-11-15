from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSequence,
)
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=1)
def get_generation_chain() -> RunnableSequence:
    """Build the generation chain on first use and reuse it."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        timeout=30.0,
        max_retries=1,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to "
                "answer the question. "
                "If you don't know the answer, just say that you "
                "don't know. "
                "Use three sentences maximum and keep the answer "
                "concise.\n"
                "Question: {question}\n"
                "Context: {context}\n"
                "Answer:",
            )
        ]
    )

    return prompt | llm | StrOutputParser()


def _invoke_generation_chain(
    inputs: dict,
    config: RunnableConfig,
) -> str:
    return get_generation_chain().invoke(inputs, config=config)


generation_chain = RunnableLambda(_invoke_generation_chain)