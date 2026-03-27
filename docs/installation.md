# Installation and Configuration Guide

## Overview

This guide walks through installation and configuration of the AI STIG Compliance Assistant.

The solution uses:

- FastAPI as the application API
- Qdrant as the vector database
- Memcached as a lightweight cache
- Ollama as the local reasoning engine
- Open WebUI as the user interface

---

## 1. System prerequisites

Recommended:
- Linux workstation
- Python 3.10+
- Docker installed
- 16 GB RAM or more
- Optional GPU for better local LLM performance

---

## 2. Clone the repository

```bash
git clone https://github.com/auslanderuz/stig-ai-assistant.git
cd stig-ai-assistant
