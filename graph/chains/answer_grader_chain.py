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

class GardeAnswer(BaseModel):
    """Binary score for answer correctness."""

    binary_score: str = Field(
        description="Yes if the answer is correct; otherwise No."
    )

structured_llm_grader = llm.with_structured_output(GardeAnswer)

system="""You are a grader assessing whether an LLM generation is correct or not. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is correct; otherwise 'no'."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User Question: {question} \n\n LLM Generation: {generation}"),
    ]
)
answer_grader= answer_prompt | structured_llm_grader