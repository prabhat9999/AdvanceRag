from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState


class WebSearchError(RuntimeError):
    """Raised when the web-search service cannot complete a request."""


@lru_cache(maxsize=1)
def get_web_search_tool() -> TavilySearch:
    """Create the search client on first use and reuse it."""
    return TavilySearch(
        max_results=3,
        search_type="web",
        search_engine="google",
    )


def web_search(state: GraphState) -> dict[str, Any]:
    """Search the web and return documents without mutating input state."""
    question = state["question"]
    existing_documents = state.get("documents") or []

    response = get_web_search_tool().invoke({"query": question})

    if "error" in response:
        raise WebSearchError(
            "Web search is unavailable. Please try again."
        )

    results = response["results"]
    new_documents = []

    if results:
        combined_content = "\n".join(
            result["content"] for result in results
        )
        new_documents.append(
            Document(page_content=combined_content)
        )

    return {
        "documents": [*existing_documents, *new_documents],
        "question": question,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    result = web_search(
        {
            "question": "agent memory",
            "documents": [],
        }
    )
    print(result)