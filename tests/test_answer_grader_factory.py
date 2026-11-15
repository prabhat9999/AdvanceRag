import importlib
from unittest.mock import Mock

from langchain_core.runnables import RunnableLambda


def test_answer_grader_builds_once_and_handles_each_request(
    monkeypatch,
):
    module = importlib.import_module(
        "graph.chains.answer_grader_chain"
    )
    module.get_answer_grader.cache_clear()

    received_prompts = []
    decisions = iter([True, False])

    def fake_grade(prompt):
        received_prompts.append(prompt.to_string())
        return module.GradeAnswer(
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

        useful = module.answer_grader.invoke(
            {
                "question": "What is agent memory?",
                "generation": "Agent memory stores past interactions.",
            }
        )
        not_useful = module.answer_grader.invoke(
            {
                "question": "What is agent memory?",
                "generation": "Pizza contains cheese.",
            }
        )

        fake_constructor.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
            timeout=30.0,
            max_retries=1,
        )
        fake_client.with_structured_output.assert_called_once_with(
            module.GradeAnswer
        )

        assert len(received_prompts) == 2
        assert all(
            "What is agent memory?" in prompt
            for prompt in received_prompts
        )
        assert "Agent memory stores past interactions." in received_prompts[0]
        assert "Pizza contains cheese." in received_prompts[1]

        assert isinstance(useful, module.GradeAnswer)
        assert useful.binary_score is True
        assert not_useful.binary_score is False
    finally:
        module.get_answer_grader.cache_clear()