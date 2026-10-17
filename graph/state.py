
from typing import TypedDict, NotRequired
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: User question
        generation: LLM-generated answer
        web_search: Whether web search is needed
        documents: List of retrieved documents
    """
    question: str
    generation: NotRequired[str]
    web_search: NotRequired[bool]
    documents: NotRequired[list[Document]]
