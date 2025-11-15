from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    # API provider: 'mistral_api' (par défaut ici)
    provider: str = "mistral_api"

    # Mistral Cloud
    mistral_api_key: str | None = None
    mistral_model_name: str = "mistral-medium"  # tiny | small | medium

    # Chemins
    schema_path: str = "docs/schemas/medical_record_schema.json"
    outputs_dir: str = "outputs"

    # Génération
    temperature: float = 0.0
    max_new_tokens: int = 1200

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Micro-check pour le dossier outputs
Path(settings.outputs_dir).mkdir(parents=True, exist_ok=True)
