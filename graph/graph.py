from typing import Literal

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader_chain import answer_grader
from graph.chains.hallucination_grader_chain import hallucination_grader
from graph.chains.router_chain import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.routing import decide_to_generate
from graph.state import GraphState


def grade_generation_grounded_in_documents_and_question(
    state: GraphState,
) -> Literal["useful", "not useful", "not supported"]:
    """Check whether the answer is grounded and addresses the question."""

    if "documents" not in state:
        raise ValueError(
            "Documents must be present before grading the generated answer."
        )

    if "generation" not in state:
        raise ValueError(
            "A generated answer must be present before answer grading."
        )

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    print("---CHECK HALLUCINATIONS---")

    grounding_score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    if grounding_score.binary_score is not True:
        print("---DECISION: GENERATION IS NOT GROUNDED---")
        return "not supported"

    print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
    print("---GRADE GENERATION vs QUESTION---")

    answer_score = answer_grader.invoke(
        {
            "question": question,
            "generation": generation,
        }
    )

    if answer_score.binary_score is True:
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"

    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
    return "not useful"


def route_question(
    state: GraphState,
) -> Literal["websearch", "retrieve"]:
    """Choose public web search or document retrieval."""

    print("---ROUTE QUESTION---")

    source: RouteQuery = question_router.invoke(
        {"question": state["question"]}
    )

    if source.datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"

    if source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"

    raise ValueError(
        f"Unsupported routing destination: {source.datasource!r}"
    )


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": RETRIEVE,
        "not supported": RETRIEVE,
    },
)

app = workflow.compile()