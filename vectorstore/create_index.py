from sqlalchemy import text

from database.postgres_sql import get_engine


def create_index(
    embedding_dimensions: int = 1536
) -> None:
    """Enable pgvector and create the document chunk table and index."""
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY,
                company VARCHAR(100) NOT NULL,
                year VARCHAR(10) NOT NULL,
                source_file TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({embedding_dimensions}) NOT NULL
            )
        """))
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """))

    print("pgvector document table created successfully.")


if __name__ == "__main__":
    create_index()