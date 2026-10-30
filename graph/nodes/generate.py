from typing import Any, Dict

from graph.chains.generation_chain import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate an answer and increment the generation counter."""

    if "documents" not in state:
        raise ValueError("Documents must be present before generation.")

    question = state["question"]
    documents = state["documents"]
    attempts = state.get("generation_attempts", 0)

    generation = generation_chain.invoke(
        {
            "question": question,
            "context": documents,
        }
    )

    return {
        "generation": generation,
        "documents": documents,
        "question": question,
        "generation_attempts": attempts + 1,
    }