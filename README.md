# AI STIG Compliance Assistant

A self-hosted AI compliance assistant that ingests DISA STIG data, stores embeddings in Qdrant, and uses Ollama + FastAPI to generate risk-based recommendations through an OpenAI-compatible API.

## Stack
- FastAPI
- Qdrant
- Memcached
- Ollama
- Open WebUI
- sentence-transformers

## Features
- STIG XML ingestion
- Semantic retrieval
- Hybrid output: structured controls + AI reasoning
- OpenAI-compatible API for Open WebUI integration

## Run
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
