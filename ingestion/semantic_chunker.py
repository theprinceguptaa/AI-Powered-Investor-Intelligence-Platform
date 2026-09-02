from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()



def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.

    Args:
        markdown_file: Markdown file path.

    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embedding_model=None
) -> list[Document]:
    """
    Generate chunks from markdown using SemanticChunker.

    Args:
        markdown_file: Markdown file path.
        embedding_model: Embeddings model for semantic chunking.

    Returns:
        List of semantic chunks.
    """
    
    markdown_content = read_markdown(markdown_file)
    
    if embedding_model is None:
        # Update the model name and fix 'task' to 'task_type'
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    splitter = SemanticChunker(embeddings = embedding_model, breakpoint_threshold_type="percentile")
    splits = splitter.split_text(markdown_content)
    return [Document(page_content=split) for split in splits]

if __name__ == "__main__":
    # Construct the path relative to this file's location
    script_dir = Path(__file__).parent
    markdown_file = script_dir / ".." / "data" / "markdown" / "2024_Apple.md"
    markdown_file = markdown_file.resolve()  # Resolve to absolute path

    chunks = chunk_markdown(
        markdown_file=str(markdown_file),
        embedding_model=None
    )

    print(f"Generated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()