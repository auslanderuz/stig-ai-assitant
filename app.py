from fastapi import FastAPI, Request, Query
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import memcache
import hashlib
import re
import html
import requests
import os

app = FastAPI()

# =========================
# CONFIG
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "stig-ai")
TOP_K = int(os.getenv("TOP_K", "20"))
AI_CONTEXT_K = int(os.getenv("AI_CONTEXT_K", "5"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "security")

MEMCACHED_HOST = os.getenv("MEMCACHED_HOST", "127.0.0.1")
MEMCACHED_PORT = int(os.getenv("MEMCACHED_PORT", "11211"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# =========================
# INIT
# =========================
qdrant = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
mc = memcache.Client([f"{MEMCACHED_HOST}:{MEMCACHED_PORT}"])
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# VECTOR SEARCH
# =========================
def search_qdrant(query: str):
    vector = model.encode(query).tolist()

    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=TOP_K
    ).points

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "score": getattr(hit, "score", 0.0),
                "severity": payload.get("severity", ""),
                "platform": payload.get("platform", ""),
                "rule_id": payload.get("rule_id", ""),
            }
        )

    return results


def filter_relevant(results, threshold=RELEVANCE_THRESHOLD):
    filtered = [r for r in results if r["score"] >= threshold]
    return filtered if filtered else results[:AI_CONTEXT_K]


# =========================
# TEXT PARSING
# =========================
def strip_xml(text: str) -> str:
    text = html.unescape(text)
    return re.sub(r"<[^>]+>", "", text)


def extract_field(text: str, label: str) -> str:
    match = re.search(rf"{label}:\s*(.*)", text)
    return match.group(1).strip() if match else ""


def extract_full_fix(text: str) -> str:
    match = re.search(r"Fix:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_registry(fix_text: str):
    hive = re.search(r"Registry Hive:\s*(.*)", fix_text)
    path = re.search(r"Registry Path:\s*(.*)", fix_text)
    value = re.search(r"Value Name:\s*(.*)", fix_text)
    configured = re.search(r"Value:\s*(.*)", fix_text)

    if hive and path:
        hive_val = hive.group(1).strip()
        path_val = path.group(1).strip()
        configured_val = configured.group(1).strip() if configured else "1"

        if value:
            value_val = value.group(1).strip()
            return (
                f"{hive_val}\\{path_val}\n"
                f"Value: {value_val} = {configured_val}"
            )
        return f"{hive_val}\\{path_val}"

    return None


def clean_stig_text(text: str):
    cleaned = strip_xml(text)

    rule_id = extract_field(cleaned, "Rule ID")
    severity = extract_field(cleaned, "Severity").upper()
    title = extract_field(cleaned, "Title")
    fix_full = extract_full_fix(cleaned)

    registry = extract_registry(fix_full)

    return {
        "rule_id": rule_id,
        "severity": severity,
        "title": title,
        "fix": registry or (fix_full[:400] if fix_full else "Refer to STIG guidance."),
    }


# =========================
# OUTPUT FORMATTING
# =========================
def build_risk_summary(parsed_items):
    high = sum(1 for p in parsed_items if p["severity"] == "HIGH")
    medium = sum(1 for p in parsed_items if p["severity"] == "MEDIUM")
    low = sum(1 for p in parsed_items if p["severity"] == "LOW")

    return f"""
📊 Risk Breakdown:
🔴 High: {high}
🟠 Medium: {medium}
🟢 Low: {low}
""".strip()


def format_controls(parsed_items):
    blocks = []

    for p in parsed_items[:3]:
        icon = {
            "HIGH": "🔴",
            "MEDIUM": "🟠",
            "LOW": "🟢"
        }.get(p["severity"], "⚪")

        blocks.append(
            f"""
{icon} {p['severity']} – {p['title']}

🛠 Recommended Action:
{p['fix']}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


# =========================
# LLM REASONING
# =========================
def call_llm(prompt: str):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        if response.status_code != 200:
            return None

        data = response.json()
        return data.get("response", "").strip() or None

    except Exception:
        return None


def build_ai_summary(query: str, context_texts):
    joined = "\n\n".join(context_texts[:AI_CONTEXT_K])

    prompt = f"""
You are a cybersecurity compliance expert.

User query:
{query}

Relevant STIG controls:
{joined}

Provide:
- Key risks
- Top 3 recommended actions
- Most critical controls

Be concise, accurate, and structured.
"""

    return call_llm(prompt)


# =========================
# CORE RESPONSE
# =========================
def process_query(query: str):
    cache_key = "v16_" + hashlib.md5(query.encode()).hexdigest()

    try:
        cached = mc.get(cache_key)
        if cached:
            return cached
    except Exception:
        pass

    raw_results = search_qdrant(query)
    relevant_results = filter_relevant(raw_results)

    context_texts = [r["text"] for r in relevant_results]
    parsed = [clean_stig_text(t) for t in context_texts]

    ai_summary = build_ai_summary(query, context_texts)
    if not ai_summary:
        ai_summary = "AI analysis unavailable."

    response = f"""
🔎 Query: {query}

📊 Summary:
Found {len(relevant_results)} relevant STIG controls.

{build_risk_summary(parsed)}

🧠 AI Analysis:
{ai_summary}

🛠 Top Controls:
{format_controls(parsed)}
""".strip()

    try:
        mc.set(cache_key, response, time=CACHE_TTL)
    except Exception:
        pass

    return response


# =========================
# DEBUG ENDPOINTS
# =========================
@app.get("/")
def root():
    return {"status": "Hybrid AI mode active"}


@app.get("/ask")
def ask(q: str = Query(...)):
    return {"answer": process_query(q)}


# =========================
# OPENAI-COMPATIBLE API
# =========================
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "custom"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    user_input = messages[-1]["content"] if messages else ""

    answer = process_query(user_input)

    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {}
    }
