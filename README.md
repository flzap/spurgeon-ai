# Spurgeon Study AI — Local RAG & Script Pipeline

A local, offline Retrieval-Augmented Generation (RAG) agent that indexes Charles H. Spurgeon's 1855 sermons (Volume 1) to answer theological queries and generate voice-ready video scripts.

Built with **LangChain**, **ChromaDB**, and **Ollama (Llama 3.1)**.

---

## Features

* **100% Local Inference:** Zero API costs and complete data privacy via Ollama (`llama3.1` + `all-minilm`).
* **Metadata Injection:** Automatically prepends sermon numbers, scripture references, dates, and titles to retrieved chunks to prevent context loss.
* **Hallucination Guardrails:** Enforces refusal rules on out-of-domain/anachronistic queries while maintaining identity awareness.
* **Script Generator:** Produces cinematic scripts formatted with visual prompts (Leonardo/Midjourney) and prosody-tuned punctuation for ElevenLabs TTS.

---

## Tech Stack

* **Framework:** LangChain
* **Vector Store:** ChromaDB (~2,100 chunks)
* **Embeddings:** `all-minilm`
* **LLM:** Llama 3.1 8B (via Ollama)
* **Language:** Python 3.10+

---

## Quickstart

### 1. Prerequisites

```bash
ollama run llama3.1
ollama pull all-minilm

```

### 2. Setup & Install

```bash
git clone https://github.com/your-username/spurgeon-rag.git
cd spurgeon-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Ingest & Run

```bash
# Build vector store from raw transcripts
python ingest.py

# Launch interactive CLI
python app.py

```

---

## Core Component: Context Injection

Passes structured metadata directly into the retrieval context to anchor implicit text references:

```python
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[SERMON #{d.metadata.get('sermon_no')} | REF: {d.metadata.get('bible_ref')} | DATE: {d.metadata.get('date')}]\n{d.page_content}"
        for d in docs
    )

```

---

## License

MIT License. Historical sermon sources are in the public domain.
