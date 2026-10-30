from langchain_core.documents import Document
from unittest.mock import Mock

from ingestion import make_chunk_id, store_documents



def test_same_source_and_text_produce_same_id():
    first = Document(
        page_content="An agent can break a task into smaller steps.",
        metadata={"source": "https://example.com/agents"},
    )
    second = Document(
        page_content="An agent can break a task into smaller steps.",
        metadata={"source": "https://example.com/agents"},
    )

    assert make_chunk_id(first) == make_chunk_id(second)


def test_changed_text_produces_different_id():
    original = Document(
        page_content="Original passage.",
        metadata={"source": "https://example.com/agents"},
    )
    changed = Document(
        page_content="Updated passage.",
        metadata={"source": "https://example.com/agents"},
    )

    assert make_chunk_id(original) != make_chunk_id(changed)


def test_different_sources_produce_different_ids():
    first = Document(
        page_content="The same passage.",
        metadata={"source": "https://example.com/first"},
    )
    second = Document(
        page_content="The same passage.",
        metadata={"source": "https://example.com/second"},
    )

    assert make_chunk_id(first) != make_chunk_id(second)

def test_existing_chunk_is_not_added_again():
    chunk = Document(
        page_content="An existing passage.",
        metadata={"source": "https://example.com/agents"},
    )

    fake_store = Mock()
    fake_store.get.return_value = {
        "ids": [make_chunk_id(chunk)],
    }

    store_documents([chunk], vectorstore=fake_store)

    fake_store.add_documents.assert_not_called()

def test_new_chunk_is_added():
    chunk = Document(
        page_content="A new passage.",
        metadata={"source": "https://example.com/agents"},
    )

    fake_store = Mock(spec_set=["get", "add_documents"])
    fake_store.get.return_value = {"ids": []}

    store_documents([chunk], vectorstore=fake_store)

    fake_store.add_documents.assert_called_once_with(
        documents=[chunk],
        ids=[make_chunk_id(chunk)],
    )

def test_mixed_batch_adds_only_new_chunk():
    existing_chunk = Document(
        page_content="An existing passage.",
        metadata={"source": "https://example.com/agents"},
    )
    new_chunk = Document(
        page_content="A new passage.",
        metadata={"source": "https://example.com/agents"},
    )

    fake_store = Mock(spec_set=["get", "add_documents"])
    fake_store.get.return_value = {
        "ids": [make_chunk_id(existing_chunk)],
    }

    store_documents(
        [existing_chunk, new_chunk],
        vectorstore=fake_store,
    )

    fake_store.add_documents.assert_called_once_with(
        documents=[new_chunk],
        ids=[make_chunk_id(new_chunk)],
    )
    #uv run python -m pytest tests/test_ingestion.py -q