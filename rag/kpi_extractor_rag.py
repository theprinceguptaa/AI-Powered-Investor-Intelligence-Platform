import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, field_validator, Field

from llm.openai import get_structured_completion
from vectorstore.pgvector import PgVectorStore, Retriever

load_dotenv()


class FinancialMetrics(BaseModel):
    revenue: str | None = Field(None, alias="Revenue with currency sign")
    net_income: str | None = Field(None, alias="Net Income")
    operating_income: str | None = Field(None, alias="Operating Income")
    cash_flow: str | None = Field(None, alias="Cash Flow from Operating Activities")
    total_assets: str | None = Field(None, alias="Total Assets")
    total_liabilities: str | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")


def retrieve_context(
    retriever: Retriever,
    company: str,
    year: int
) -> str:
    """
    Retrieve broad financial context from the vector store.
    """
    query = f"""
    Annual report financial statements,
    income statement,
    balance sheet,
    cash flow statement,
    risks,
    growth drivers,
    financial performance
    for {company} fiscal year {year}
    """

    documents = retriever.invoke(
        query=query,
        company=company,
        year=year,
        top_k=20
    )
    # print(documents)
    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build KPI extraction prompt.
    """
    return f"""
You are an expert financial analyst.

Company: {company}
Year: {year}

Context:
{context}

Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities
5. Total Assets
6. Total Liabilities
7. Top Risk Factors
8. Top Growth Drivers

Instructions:

- Use only the provided context.
- Return null if unavailable.
- Financial values must match the report exactly.
- Preserve the currency used in the report.
- Always include the currency symbol in every financial value.
- Always include the source unit when present, such as thousand, million, billion, lakh, or crore.
- For INR use ₹ and Indian numbering units: thousand, lakh, crore.
- For USD use $ and international units: thousand, million, billion.
- Never invent a currency or unit. Return null if the report does not identify it.
- Return values as strings, for example:
  "$391.04 billion"
  "₹97,690 crore"
  "₹12.5 lakh"
  "$500 million"
- Risk factors should be concise.
- Growth drivers should be concise.
- Return valid JSON only.
"""


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int
) -> dict:
    """
    Extract KPIs using RAG.
    """
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year
    )

    prompt = build_extraction_prompt(
        company=company,
        year=year,
        context=context
    )

    metrics = get_structured_completion(
        prompt=prompt,
        response_model=FinancialMetrics
    )

    return metrics.model_dump()


def main() -> None:
    company = "Apple"
    year = 2024

    embedding_model = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    vector_store = PgVectorStore()

    retriever = Retriever(
        vector_store,
        embedding_model
    )

    results = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=year
    )

    print(f"\nExtracted KPIs for {company} {year}\n")

    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)


    from database.save_metrics import save_metrics

    save_metrics(
        company=company,
        year=year,
        metrics=results
    )

if __name__ == "__main__":
    main()