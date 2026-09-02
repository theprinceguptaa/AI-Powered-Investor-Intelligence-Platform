import shutil
from fastapi import APIRouter, File, UploadFile
from pathlib import Path
import os
from langchain_openai import OpenAIEmbeddings
from vectorstore.pgvector import PgVectorStore
from ingestion.ingest_documents import ingest_document

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    upload_dir = Path("data/raw_pdfs")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

        vector_store = PgVectorStore()

        ingest_document(
            pdf_path=str(file_path),
            embedding_model=embeddings,
            vector_store=vector_store
        )

    return {
        "message": "Document uploaded successfully",
        "file_name": file.filename
    }