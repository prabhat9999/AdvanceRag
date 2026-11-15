import importlib
from unittest.mock import Mock

from langchain_core.runnables import RunnableLambda


def test_grader_client_is_created_on_demand_and_reused(monkeypatch):
    module = importlib.import_module(
        "graph.chains.hallucination_grader_chain"
    )

    module.get_hallucination_grader.cache_clear()

    fake_client = Mock()
    fake_client.with_structured_output.return_value = RunnableLambda(
        lambda inputs: module.GradeHallucinations(binary_score=True)
    )

    fake_constructor = Mock(return_value=fake_client)
    monkeypatch.setattr(module, "ChatOpenAI", fake_constructor)

    try:
        # Replacing the constructor alone shouldn't create a client.
        fake_constructor.assert_not_called()

        first = module.get_hallucination_grader()
        second = module.get_hallucination_grader()

        fake_constructor.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
            timeout=30.0,
            max_retries=1,
        )
        assert first is second
    finally:
        # Don't leave a cached fake chain for other tests.
        module.get_hallucination_grader.cache_clear()