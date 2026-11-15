import importlib
from unittest.mock import Mock

from langchain_core.runnables import RunnableLambda


def test_retrieval_grader_builds_once_and_handles_each_request(
    monkeypatch,
):
    module = importlib.import_module(
        "graph.chains.retrieval_grader_chain"
    )
    module.get_retrieval_grader.cache_clear()

    received_prompts = []
    decisions = iter([True, False])

    def fake_grade(prompt):
        received_prompts.append(prompt.to_string())
        return module.GradeDocuments(
            binary_score=next(decisions)
        )

    fake_client = Mock()
    fake_client.with_structured_output.return_value = RunnableLambda(
        fake_grade
    )

    fake_constructor = Mock(return_value=fake_client)
    monkeypatch.setattr(module, "ChatOpenAI", fake_constructor)

    try:
        fake_constructor.assert_not_called()

        relevant = module.retrieval_grader.invoke(
            {
                "question": "What is agent memory?",
                "document": "Agent memory stores past interactions.",
            }
        )
        irrelevant = module.retrieval_grader.invoke(
            {
                "question": "How do I make pizza?",
                "document": "Agent memory stores past interactions.",
            }
        )

        fake_constructor.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
        )
        fake_client.with_structured_output.assert_called_once_with(
            module.GradeDocuments
        )

        assert len(received_prompts) == 2
        assert "What is agent memory?" in received_prompts[0]
        assert "How do I make pizza?" in received_prompts[1]
        assert all(
            "Agent memory stores past interactions." in prompt
            for prompt in received_prompts
        )

        assert isinstance(relevant, module.GradeDocuments)
        assert relevant.binary_score is True
        assert irrelevant.binary_score is False
    finally:
        module.get_retrieval_grader.cache_clear()