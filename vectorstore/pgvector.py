import uuid
from types import SimpleNamespace

from sqlalchemy import text

from database.postgres_sql import get_engine


class PgVectorStore:
    """PostgreSQL vector store backed by the pgvector extension."""

    def __init__(self, embedding_dimensions: int = 1536) -> None:
        self.engine = get_engine()
        self.embedding_dimensions = embedding_dimensions
        self.client = self

    def upload_chunks(
        self,
        chunks,
        embedding_model,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        """Embed and persist document chunks in PostgreSQL."""
        documents = [
            {
                "id": str(uuid.uuid4()),
                "company": company,
                "year": year,
                "source_file": source_file,
                "content": chunk.page_content,
                "embedding": _as_vector(embedding_model.embed_query(chunk.page_content))
            }
            for chunk in chunks
        ]

        if documents:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO document_chunks
                            (id, company, year, source_file, content, embedding)
                        VALUES
                            (:id, :company, :year, :source_file, :content,
                             CAST(:embedding AS vector))
                        """
                    ),
                    documents
                )

        print(f"Uploaded {len(documents)}/{len(documents)} chunks.")

    def search(
        self,
        query: str,
        query_embedding,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """Return the nearest document chunks using cosine distance."""
        conditions = []
        params = {
            "query_embedding": _as_vector(query_embedding),
            "top_k": top_k
        }
        if company:
            conditions.append("company = :company")
            params["company"] = company
        if year is not None:
            conditions.append("year = :year")
            params["year"] = str(year)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query_text = text(
            f"""
            SELECT content
            FROM document_chunks
            {where_clause}
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
            """
        )

        with self.engine.connect() as connection:
            results = connection.execute(query_text, params)
            return [SimpleNamespace(page_content=row.content) for row in results]


class Retriever:
    """Retriever compatible with the application's existing RAG callers."""

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        query_embedding = self.embedding_model.embed_query(query)
        return self.vector_store.search(
            query=query,
            query_embedding=query_embedding,
            company=company,
            year=year,
            top_k=top_k
        )


def _as_vector(values) -> str:
    """Format an embedding for pgvector's text input format."""
    return "[" + ",".join(str(value) for value in values) + "]"