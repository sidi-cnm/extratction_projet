# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.extract import router as extract_router
from .routers.index import router as index_router  # 👈 importer le router index

app = FastAPI(
    title="Medical Doc Extract API",
    version="1.0.0",
    description="Extraction structurée (JSON) depuis documents médicaux via Mistral API",
)

# CORS (ouvre si tu as un front web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": [
            "/extract",
            "/extract-file",
            "/index-json",
            "/docs",
        ],
    }

# 👉 Enregistrer les routers ici
app.include_router(extract_router)
app.include_router(index_router)   # 👈 maintenant /index-json est connu de FastAPI
