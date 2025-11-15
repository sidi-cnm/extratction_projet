# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ExtractRequest(BaseModel):
    text: str = Field(..., description="Texte brut extrait du PDF/OCR")

class ExtractFileResponse(BaseModel):
    filename: str
    size: int
    content_type: str
    # champ interne json_ mais on accepte/renvoit "json" via alias
    json_: Dict[str, Any] = Field(..., alias="json")
    valid: bool = True
    validation_error: Optional[str] = None

    # Permet de construire le modèle en passant soit json soit json_
    model_config = {
        "populate_by_name": True
    }

class ExtractResponse(BaseModel):
    json_: Dict[str, Any] = Field(..., alias="json")
    valid: bool = True
    validation_error: Optional[str] = None

    model_config = {
        "populate_by_name": True
    }
