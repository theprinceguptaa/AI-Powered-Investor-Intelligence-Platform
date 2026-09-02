import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from vectorstore.pgvector import PgVectorStore, Retriever


def search_vectorstore(query: str, top: int = 5):
    embedding_model = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    store = PgVectorStore()
    results = Retriever(store, embedding_model).invoke(query, top_k=top)

    print(f"Query: {query!r}")
    print(f"Top: {top}")
    print(f"Results: {len(results)}\n")

    for idx, result in enumerate(results, start=1):
        content = result.page_content

        snippet = content.strip().replace("\n", " ") if isinstance(content, str) else "<no content>"
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"Result {idx}")
        print(f"  content snippet: {snippet}")
        print("  " + "-" * 60)

    if not results:
        print("No results returned. Verify your index contents or try a different query.")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m rag.retrieval_debug \"your query here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_vectorstore(query)


if __name__ == "__main__":
    main()
