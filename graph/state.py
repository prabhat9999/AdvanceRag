from typing import NotRequired, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """Data passed between graph nodes."""

    question: str
    generation: NotRequired[str]
    web_search: NotRequired[bool]
    documents: NotRequired[list[Document]]
    generation_attempts: NotRequired[int]