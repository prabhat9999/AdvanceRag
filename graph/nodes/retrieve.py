from typing import Any,Dict
from ingestion import retriever
from graph.state import GraphState


def retrieve(state:GraphState)->Dict[str,Any]:
    """
  Retrieves the question and get the documents from the state.
    """
    print("--Retrieve--")
    question=state["question"]
    documents=retriever.invoke(question)
    return {"question":question,"documents":documents}

