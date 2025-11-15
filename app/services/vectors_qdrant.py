import os, uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()  # par défaut, il cherche un fichier .env à la racine
QDRANT_HOST = os.getenv("QDRANT_HOST")
print("QDRANT_HOST =", QDRANT_HOST)
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_records")
API_KEY= os.getenv("api_key")
print("API_KEY" , API_KEY)
client = QdrantClient(url=QDRANT_HOST, api_key=API_KEY)

def ensure_collection(vector_size: int):
    # recrée la collection si elle n'existe pas avec la bonne dimension
    try:
        info = client.get_collection(COLLECTION)
        # si la taille ne correspond pas, on la recrée
        # (on simplifie: on recrée toujours)
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    except Exception:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def upsert_passages(doc_id: str, texts: List[str], vectors: List[List[float]], metas: List[Dict], raw_json: Dict):
    points = []
    for i, (vec, text, meta) in enumerate(zip(vectors, texts, metas)):
        pid = int(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{i}").int % (10**12))
        payload = {
            "doc_id": doc_id,
            "text": text,
            "section": meta.get("section"),
            "meta": meta,
            "raw_json": raw_json,
            "provider": "mistral",
            "embed_model": os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
        }
        points.append(PointStruct(id=pid, vector=vec, payload=payload))
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)

def knn_search(query_vec: List[float], top_k: int = 5, doc_id: str | None = None):
    flt = None
    if doc_id:
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    res = client.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter=flt
    )
    return res
