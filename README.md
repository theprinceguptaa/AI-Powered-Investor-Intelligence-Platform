# AI-Powered Investor Intelligence Platform


This repository contains the Python backend for an AI-powered Investor Intelligence Platform, including document ingestion, pgvector semantic search, KPI extraction, Azure OpenAI integration, and PostgreSQL-based KPI storage.

## Prerequisites

* Python 3.12+
* UV Package Manager

## Setup

### 1. Install UV

#### Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### macOS/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:

```bash
uv --version
```

---

### 2. Create Virtual Environment

```bash
uv venv
```

---

### 3. Activate Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

---

### 4. Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

### 5. Configure Environment Variables

Create a `.env` file and configure all required environment variables before running the application.

For PostgreSQL with pgvector, configure at least:

```text
POSTGRES_HOST=your-project.pooler.supabase.com
POSTGRES_PORT=6543
POSTGRES_DATABASE=postgres
POSTGRES_USER=postgres.your-project-ref
POSTGRES_PASSWORD=your-database-password
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Use the database password from Supabase, not the account login password. Keep `.env` out of source control.

---

### 6. Run the Application

```bash
python main.py
```

---

## Project Features

* Annual Report Upload & Processing
* KPI Extraction using OpenAI
* PostgreSQL pgvector Integration
* Semantic Search & Retrieval
* RAG-based Chatbot
* PostgreSQL KPI Storage
* Investor Insights Dashboard
* Production-Grade Modular Architecture

---

## Technology Stack

### Backend

* FastAPI
* Python 3.13

### AI Services

* OpenAI
* PostgreSQL with pgvector

### Database

*  PostgreSQL

### Package Management

* UV

