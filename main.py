import argparse

from dotenv import load_dotenv
from openai import APITimeoutError

from config import validate_environment


def parse_args(argv=None):
    """Read and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ask a question using the AdvanceRag workflow."
    )

    parser.add_argument(
        "question",
        help="The question you want answered. Use quotes around it.",
    )

    args = parser.parse_args(argv)

    if not args.question.strip():
        parser.error("question must not be empty")

    return args


def main(argv=None):
    args = parse_args(argv)

    load_dotenv()

    try:
        validate_environment(
            ("OPENAI_API_KEY", "TAVILY_API_KEY")
        )
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None

    from graph.graph import app

    print("Advanced RAG")

    try:
        result = app.invoke(
            input={"question": args.question.strip()}
        )
    except APITimeoutError:
        raise SystemExit(
            "Service error: the model request timed out. "
            "Please try again."
        ) from None

    print("Final Output:", result["generation"])


if __name__ == "__main__":
    main()