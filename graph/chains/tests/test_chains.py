from typing import cast

from dotenv import load_dotenv
from ingestion import retriever
from graph.chains.retrieval_grader_chain import GradeDocuments,retrieval_grader
from graph.chains.generation_chain import generation_chain
from graph.chains.hallucination_grader_chain import GradeHallucinations,hallucination_grader
from graph.chains.router_chain import question_router, RouteQuery

load_dotenv()

def test_retrieval_grader_yes()->None:
        question = "agent memory"
        docs = retriever.invoke(question)
        doc_txt = docs[0].page_content
        res = cast(GradeDocuments, retrieval_grader.invoke({"question": question, "document": doc_txt}))
        assert res.binary_score == "yes"

   

def test_retrieval_grader_no()->None:
          question = "pizza"
          docs = retriever.invoke(question)
          doc_txt = docs[0].page_content
          res = cast(GradeDocuments, retrieval_grader.invoke({"document":doc_txt,"question":question}))
          assert res.binary_score == "no"


def test_generation_chain()->None:
        question="Agent memory"
        docs = retriever.invoke(question)
        generation=generation_chain.invoke({"question":question,"context":docs})
        print(generation)


def test_hallucination_grader_answer_yes() -> None:
        question="Agent memory"
        docs=retriever.invoke(question)
        generation=generation_chain.invoke({"question":question,"context":docs})
        res=cast(GradeHallucinations,hallucination_grader.invoke({"documents":docs,"generation":generation}))
        assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
        question="what is pizza"
        docs=retriever.invoke(question)
        generation=generation_chain.invoke({"question":question,"context":docs})
        res=cast(GradeHallucinations,hallucination_grader.invoke({"documents":docs,"generation":generation}))
        assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"