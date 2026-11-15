import importlib
from unittest.mock import Mock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


def test_generation_chain_builds_once_and_handles_each_request(
    monkeypatch,
):
    module = importlib.import_module(
        "graph.chains.generation_chain"
    )
    module.get_generation_chain.cache_clear()

    # A fake model receives the formatted prompt and returns a message.
    received_prompts = []

    def fake_model_response(prompt):
        received_prompts.append(prompt.to_string())
        return AIMessage(content="Agent memory stores information.")

    fake_constructor = Mock(
        return_value=RunnableLambda(fake_model_response)
    )
    monkeypatch.setattr(module, "ChatOpenAI", fake_constructor)

    try:
        fake_constructor.assert_not_called()

        first = module.generation_chain.invoke(
            {
                "question": "What is agent memory?",
                "context": "Memory stores information.",
            }
        )
        second = module.generation_chain.invoke(
            {
                "question": "Why is memory useful?",
                "context": "Memory supports future decisions.",
            }
        )

        # Client construction happens once.
        fake_constructor.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
            timeout=30.0,
            max_retries=1,
        )

        # Both requests still reach the model.
        assert len(received_prompts) == 2
        assert "What is agent memory?" in received_prompts[0]
        assert "Memory stores information." in received_prompts[0]
        assert "Why is memory useful?" in received_prompts[1]
        assert "Memory supports future decisions." in received_prompts[1]

        # The output parser converts the model message to plain text.
        assert first == "Agent memory stores information."
        assert second == "Agent memory stores information."
    finally:
        module.get_generation_chain.cache_clear()