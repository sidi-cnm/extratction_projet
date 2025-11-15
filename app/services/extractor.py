import os, json, re
from datetime import datetime
from pathlib import Path
import pdfplumber
from jsonschema import validate, ValidationError
from mistralai import Mistral
from ..settings import settings

# ─────────────────────────────
# Chargement du schéma
# ─────────────────────────────
SCHEMA = json.loads(Path(settings.schema_path).read_text(encoding="utf-8"))

# ─────────────────────────────
# Extraction texte PDF
# ─────────────────────────────
def extract_text_from_pdf(file_path: str) -> str:
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            chunks.append(t)
    return "\n".join(chunks).strip()

# ─────────────────────────────
# Prompt builder
# ─────────────────────────────
def build_prompt(text: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""
Tu es un extracteur clinique. À partir du texte source, produis STRICTEMENT un JSON conforme au schéma.

Règles strictes :
- Sortie = JSON UNIQUEMENT (aucun commentaire, aucune prose).
- N’utilise que les caractères JSON : {{ }} [ ] , : " .
- Dates au format YYYY-MM-DD ; si inconnu -> null (si mois/jour inconnus → 01 par défaut).
- N'invente rien ; si absent → null ou [].
- Respecte toutes les clés/type du schéma.

Schéma JSON :
{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}

Contraintes meta :
- meta.langue = "fr"
- meta.date_extraction = "{today}"
- meta.modele_utilise = "{settings.mistral_model_name}"
- meta.schema_version = "1.0"

Texte source :
<<<
{text}
>>>

⚠️ IMPORTANT :
- Un seul objet JSON valide, commençant par '{{' et finissant par '}}'.
- AUCUN TEXTE hors JSON.
"""

# ─────────────────────────────
# Appel Mistral Cloud
# ─────────────────────────────
def call_mistral_api(prompt: str) -> str:
    api_key = settings.mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquante (env/.env).")
    client = Mistral(api_key=api_key)
    res = client.chat.complete(
        model=settings.mistral_model_name,
        messages=[
            {"role": "system", "content": "Tu es un assistant médical spécialisé en extraction structurée. Réponds UNIQUEMENT en JSON valide."},
            {"role": "user", "content": prompt},
        ],
        temperature=settings.temperature,
    )
    return res.choices[0].message.content.strip()

# ─────────────────────────────
# Extraction JSON robuste
# ─────────────────────────────
def extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Réponse vide.")

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

    s = text
    starts = [i for i,c in enumerate(s) if c == "{"]
    for st in starts:
        depth = 0
        for j in range(st, len(s)):
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[st:j+1]
                    try:
                        return json.loads(cand)
                    except json.JSONDecodeError:
                        pass
                    break
    raise ValueError("Impossible d'extraire un JSON valide.")

# ─────────────────────────────
# Validation JSON vs schéma
# ─────────────────────────────
def validate_json(data: dict) -> tuple[bool, str | None]:
    try:
        validate(instance=data, schema=SCHEMA)
        return True, None
    except ValidationError as e:
        return False, e.message

# ─────────────────────────────
# Pipeline texte → JSON final
# ─────────────────────────────
def process_text(text: str) -> tuple[dict, bool, str | None]:
    prompt = build_prompt(text)
    raw = call_mistral_api(prompt)
    data = extract_json(raw)
    valid, err = validate_json(data)
    return data, valid, err
