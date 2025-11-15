from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ExtractRequest(BaseModel):
    text: str = Field(..., description="Texte brut extrait du PDF/OCR")

class ExtractFileResponse(BaseModel):
    filename: str
    size: int
    content_type: str
    json: Dict[str, Any]
    valid: bool = True
    validation_error: Optional[str] = None

class ExtractResponse(BaseModel):
    json: Dict[str, Any]
    valid: bool = True
    validation_error: Optional[str] = None
