from dotenv import load_dotenv
from graph.chains.generation_chain import generation_chain
from typing import Any, Dict
from graph.state import GraphState


def generate(state:GraphState) -> Dict[str, Any]:
    question = state["question"]
    docs = state["documents"]

    generation = generation_chain.invoke({"question": question, "context": docs})
    return {"generation": generation, "documents": docs, "question": question}   
