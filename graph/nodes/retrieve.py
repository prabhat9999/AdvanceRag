from typing import Any,Dict
from ingestion import get_retriever
from graph.state import GraphState


def retrieve(state:GraphState)->Dict[str,Any]:
    """Retrieve document chunks relevant to the question."""
    print("--Retrieve--")
    question=state["question"]
    retriever = get_retriever()
    documents=retriever.invoke(question)
    return {"question":question,"documents":documents}

