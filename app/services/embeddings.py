import os, requests
from typing import List
from ..settings import settings

MISTRAL_API_KEY = settings.mistral_api_key
EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
EMBED_URL = "https://api.mistral.ai/v1/embeddings"

def embed_texts_mistral(texts: List[str]) -> List[list]:
    """
    Appelle l'API Mistral Embeddings et retourne une liste de vecteurs.
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY manquante")
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL, "input": texts}
    r = requests.post(EMBED_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Mistral embeddings error {r.status_code}: {r.text}")
    data = r.json()["data"]
    # data est une liste d'objets {"object":"embedding","embedding":[...],"index":i}
    embs = [d["embedding"] for d in data]
    return embs
