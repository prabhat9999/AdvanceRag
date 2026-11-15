# AdvanceRag

A Python RAG application built with LangGraph. It routes questions
to a local document collection or web search, checks retrieved
documents for relevance, and grades generated answers for grounding
and usefulness.

Rejected answers trigger another attempt, up to a configured limit.
When that limit is reached, the application returns an
insufficient-evidence response.

## Requirements

- Python 3.11 or later
- uv: https://docs.astral.sh/uv/getting-started/installation/
- An OpenAI API key for language models and embeddings
- A Tavily API key for web search

## Local setup

Clone the repository and enter its directory:

```bash
git clone https://github.com/prabhat9999/AdvanceRag.git
cd AdvanceRag
```

Install dependencies, including development tools:

```bash
uv sync --locked --dev
```

For a new setup, copy the configuration template:

```bash
cp .env.example .env
```

If you already have a `.env` file, keep it instead of overwriting it.

Fill in your API keys in `.env`:

```dotenv
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

The `.env` file is ignored by Git. Keep real API keys out of commits.

## Ingest documents

Download, split, embed, and store the configured source articles:

```bash
uv run python ingestion.py
```

The local Chroma collection is stored in `.chroma/`.
Ingestion calls the OpenAI embeddings API and can incur usage costs.

## Run the application

```bash
uv run python main.py
```

Currently, `main.py` runs a fixed example question.
Change its `question` value to try another question.

Running the application calls external APIs and can incur usage costs.

## Run tests

Run the default test suite:

```bash
uv run python -m pytest
```

Integration tests are excluded by default. Tests using mocked
dependencies check application behavior without calling real services.

To run the integration tests explicitly:

```bash
uv run python -m pytest -m integration
```

Integration tests require valid service credentials and an ingested
document collection. They call real services and can incur usage costs.

## Continuous integration

GitHub Actions installs locked dependencies and runs the default
test suite on pushes and pull requests.

## Current limitations

- The entry point uses a fixed example question.
- Answer quality has not yet been measured against a fixed evaluation dataset.
- LLM-based grading is a useful check, but does not guarantee correctness.



## Run the application

Pass your question in quotes:

```bash
uv run python main.py "What is agent memory?"
```

For questions about the source articles, ingest the documents first.

To view command-line help:

```bash
uv run python main.py --help
```

Running a question calls external APIs and can incur usage costs.
Displaying help does not call external APIs.