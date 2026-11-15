import importlib
from unittest.mock import Mock

from langchain_core.runnables import RunnableLambda


def test_question_router_builds_once_and_handles_each_request(
    monkeypatch,
):
    module = importlib.import_module("graph.chains.router_chain")
    module.get_question_router.cache_clear()

    received_prompts = []
    destinations = iter(["vectorstore", "websearch"])

    def fake_route(prompt):
        received_prompts.append(prompt.to_string())
        return module.RouteQuery(
            datasource=next(destinations)
        )

    fake_client = Mock()
    fake_client.with_structured_output.return_value = RunnableLambda(
        fake_route
    )

    fake_constructor = Mock(return_value=fake_client)
    monkeypatch.setattr(module, "ChatOpenAI", fake_constructor)

    try:
        fake_constructor.assert_not_called()

        document_route = module.question_router.invoke(
            {"question": "What is agent memory?"}
        )
        web_route = module.question_router.invoke(
            {"question": "How do I make pizza?"}
        )

        fake_constructor.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
        )
        fake_client.with_structured_output.assert_called_once_with(
            module.RouteQuery
        )

        assert len(received_prompts) == 2
        assert "What is agent memory?" in received_prompts[0]
        assert "How do I make pizza?" in received_prompts[1]

        assert isinstance(document_route, module.RouteQuery)
        assert document_route.datasource == "vectorstore"
        assert web_route.datasource == "websearch"
    finally:
        module.get_question_router.cache_clear()