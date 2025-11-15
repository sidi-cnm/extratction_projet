from fastapi import APIRouter, HTTPException
from typing import Dict
from ..services.passage_builder import json_to_passages
from ..services.embeddings import embed_texts_mistral
from ..services.vectors_qdrant import ensure_collection, upsert_passages
import uuid  # <- importer uuid

router = APIRouter(prefix="", tags=["index"])

@router.post("/index-json")
def index_json(doc: Dict):
    """
    Reçoit un JSON médical (sortie de /extract), crée des passages,
    embed via Mistral, et upsert dans Qdrant.
    """
    try:
        # Générer un doc_id unique pour ce document
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"  # ex: doc_1a2b3c4d

        passages = json_to_passages(doc)
        texts = [t for t, _ in passages]
        metas = [m for _, m in passages]
        vectors = embed_texts_mistral(texts)

        # Crée / ajuste la collection à la bonne dimension (auto)
        ensure_collection(vector_size=len(vectors[0]))

        n = upsert_passages(doc_id=doc_id, texts=texts, vectors=vectors, metas=metas, raw_json=doc)
        return {"doc_id": doc_id, "inserted": n, "dimension": len(vectors[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
