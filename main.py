from dotenv import load_dotenv

from config import validate_environment


def main():
    # Load local settings before checking required values.
    load_dotenv()

    try:
        validate_environment(
            ("OPENAI_API_KEY", "TAVILY_API_KEY")
        )
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None

    # Import the graph after configuration passes validation.
    from graph.graph import app

    print("Advanced RAG")

    result = app.invoke(
        input={"question": "What is pizza?"}
    )

    print("Final Output:", result["generation"])


if __name__ == "__main__":
    main()