import importlib
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document


@pytest.mark.parametrize(
    "initial_attempts, expected_attempts",
    [
        (None, 1),
        (1, 2),
    ],
)
def test_generate_updates_attempt_counter(
    monkeypatch,
    initial_attempts,
    expected_attempts,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    module = importlib.import_module("graph.nodes.generate")

    fake_chain = Mock(spec_set=["invoke"])
    fake_chain.invoke.return_value = "A test answer."

    monkeypatch.setattr(module, "generation_chain", fake_chain)

    documents = [
        Document(page_content="Agents can divide tasks into smaller steps.")
    ]
    state = {
        "question": "What is task decomposition?",
        "documents": documents,
    }

    if initial_attempts is not None:
        state["generation_attempts"] = initial_attempts

    result = module.generate(state)

    assert result["generation_attempts"] == expected_attempts
    assert result["generation"] == "A test answer."
    assert result["documents"] == documents
    assert result["question"] == state["question"]

    fake_chain.invoke.assert_called_once_with(
        {
            "question": state["question"],
            "context": documents,
        }
    )