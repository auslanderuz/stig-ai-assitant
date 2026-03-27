import os
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# =========================
# CONFIG
# =========================
STIG_DIR = os.getenv("STIG_DIR", "stig")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "security")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))

# =========================
# INIT
# =========================
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_all_stig_files():
    files = []
    for root, _, filenames in os.walk(STIG_DIR):
        for filename in filenames:
            if filename.endswith(".xml") and "xccdf" in filename.lower():
                files.append(os.path.join(root, filename))
    return files


def get_child_text_by_suffix(element, suffix: str) -> str:
    for child in element:
        if child.tag.endswith(suffix):
            return "".join(child.itertext()).strip()
    return ""


def parse_stig(file_path):
    docs = []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as exc:
        print(f"Failed to parse {file_path}: {exc}")
        return docs

    platform = file_path.lower()

    for rule in root.iter():
        if not rule.tag.endswith("Rule"):
            continue

        rule_id = rule.get("id", "")
        severity = rule.get("severity", "").lower()

        title = get_child_text_by_suffix(rule, "title")
        description = get_child_text_by_suffix(rule, "description")
        fix = get_child_text_by_suffix(rule, "fixtext")

        if not title or not description:
            continue

        text = f"""
Rule ID: {rule_id}
Severity: {severity}
Platform: {platform}

Title: {title}

Description:
{description}

Fix:
{fix}
""".strip()

        docs.append(
            {
                "rule_id": rule_id,
                "severity": severity,
                "platform": platform,
                "text": text,
            }
        )

    return docs


def recreate_collection(vector_size: int):
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def ingest_docs(all_docs):
    texts = [doc["text"] for doc in all_docs]
    print(f"Embedding {len(texts)} STIG controls...")

    vectors = model.encode(texts).tolist()

    recreate_collection(len(vectors[0]))

    total = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch_docs = all_docs[i:i + BATCH_SIZE]
        batch_vectors = vectors[i:i + BATCH_SIZE]

        points = []
        for j, doc in enumerate(batch_docs):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=batch_vectors[j],
                    payload={
                        "rule_id": doc["rule_id"],
                        "severity": doc["severity"],
                        "platform": doc["platform"],
                        "text": doc["text"][:4000],
                    },
                )
            )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )

        print(f"Uploaded {i + len(points)} / {total}")

    print("Ingestion complete.")


def main():
    files = get_all_stig_files()
    print(f"Found {len(files)} STIG XML files")

    all_docs = []

    for file_path in files:
        docs = parse_stig(file_path)
        all_docs.extend(docs)

    print(f"Parsed {len(all_docs)} STIG controls")

    if not all_docs:
        print("No STIG data found. Check your XML files.")
        return

    ingest_docs(all_docs)


if __name__ == "__main__":
    main()
