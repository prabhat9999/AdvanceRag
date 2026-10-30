from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import hashlib
import json
load_dotenv()
COLLECTION_NAME="rag-chroma-v2"


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

def make_chunk_id(chunk):
    """Create a repeatable ID from a chunk's source and text."""

    identity = {
        "source": chunk.metadata["source"],
        "text": chunk.page_content,
    }

    serialized = json.dumps(identity, sort_keys=True)

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def load_and_split_documents():
    """Download source pages and split them into smaller passages."""

    docs = [WebBaseLoader(url).load() for url in urls]

    flattened_docs = []
    for sublist in docs:
        for doc in sublist:
            flattened_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )

    doc_splits = text_splitter.split_documents(flattened_docs)

    return doc_splits

def store_documents(chunks,vectorstore=None):

    """Embed and store only chunks whose IDs are not already saved."""

    unique_chunks = {}

    for chunk in chunks:
        chunk_id = make_chunk_id(chunk)
        unique_chunks[chunk_id] = chunk

    if vectorstore is None:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(),
            persist_directory="./.chroma",
        )

    existing_ids = set(vectorstore.get(include=[])["ids"])

    new_ids = [
        chunk_id
        for chunk_id in unique_chunks
        if chunk_id not in existing_ids
    ]

    if new_ids:
        new_chunks = [unique_chunks[chunk_id] for chunk_id in new_ids]

        vectorstore.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )

    print(f"Added {len(new_ids)} new chunks.")
    print(f"Skipped {len(unique_chunks) - len(new_ids)} existing chunks.")

    """Save chunks using repeatable IDs."""

    unique_chunks = {}

    for chunk in chunks:
        chunk_id = make_chunk_id(chunk)
        unique_chunks[chunk_id] = chunk

    vectorstore = Chroma.from_documents(
        documents=list(unique_chunks.values()),
        ids=list(unique_chunks.keys()),
        collection_name=COLLECTION_NAME,
        embedding=OpenAIEmbeddings(),
        persist_directory="./.chroma",
    )

    return vectorstore
def get_retriever():
    """Open the stored collection and return a retriever."""

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./.chroma",
    )

    return vectorstore.as_retriever()
if __name__ == "__main__":
    chunks = load_and_split_documents()
    print(f"Created {len(chunks)} chunks.")
    vectorstore = store_documents(chunks)
    print("Finished storing chunks in Chroma.")

    stored_documents = vectorstore.get()
    stored_count = len(stored_documents["ids"])
    print(f"Collection contains {stored_count} chunks.")
