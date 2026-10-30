from typing import Any, Dict, cast

from graph.chains.retrieval_grader_chain import GradeDocuments, retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Keep relevant documents and request web search if none remain."""

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs = []

    for d in documents:
        score = cast(
            GradeDocuments,
            retrieval_grader.invoke(
                {
                    "question": question,
                    "document": d if isinstance(d, str) else d.page_content,
                }
            ),
        )

        if score.binary_score is True:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    web_search = len(filtered_docs) == 0

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }