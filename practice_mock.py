from unittest.mock import MagicMock

fake_grader = MagicMock()

fake_grader.invoke.return_value.binary_score = False

result = fake_grader.invoke(
    {"question": "What is RAG?"}
)

print(result.binary_score)
print(fake_grader.invoke.call_count)