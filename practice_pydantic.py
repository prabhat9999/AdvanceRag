from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="True if the answer is correct"
    )


result1 = GradeAnswer(binary_score=True)
result2 = GradeAnswer(binary_score=False)

print(result1.binary_score)
print(result2.binary_score)
print(type(result1.binary_score))