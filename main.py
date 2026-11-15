from dotenv import load_dotenv


def main():
    load_dotenv()

    from graph.graph import app

    print("Advanced RAG")
    result = app.invoke(input={"question": "What is pizza?"})
    print("Final Output:", result["generation"])


if __name__ == "__main__":
    main()