from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

class GradeAnswer(BaseModel):
    """Binary score for answer correctness."""

    binary_score: bool = Field(
        description="True if the answer addresses the question; otherwise False."
    )

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system="""You are a grader assessing whether an LLM generation is correct or not. \n
     Give a binary score True or False. True means that the answer is correct; otherwise False."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User Question: {question} \n\n LLM Generation: {generation}"),
    ]
)
answer_grader= answer_prompt | structured_llm_grader