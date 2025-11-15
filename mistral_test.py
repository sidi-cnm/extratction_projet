import os
import json
import re
from datetime import datetime
from pathlib import Path

import pdfplumber
from jsonschema import validate, ValidationError
from mistralai import Mistral

# (optionnel) .env pour la clé
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
# Clé API (priorité à l'environnement)
MISTRAL_API_KEY = "2gFcRaKLzhzz8bHVK0pykVLTeD3jMVLO"

# Modèle cloud : tiny | small | medium (selon précision/coût)
MODEL_NAME =  "mistral-medium"

# Chemins
SCHEMA_PATH = "docs/schemas/medical_record_schema.json"
PDF_PATH = "data/file1.pdf"

# Dossiers sorties
OUT_DIR = Path("outputs")
LOGS_DIR = OUT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Extraction de texte PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extraction simple avec pdfplumber.
    (Pour les PDFs scannés/vides, ajoute un fallback OCR plus tard.)
    """
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts).strip()

# ─────────────────────────────────────────────
# 3. Construction du prompt (avec schéma)
# ─────────────────────────────────────────────
def build_prompt(text: str, schema: dict) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""
Tu es un extracteur clinique. À partir du texte source, produis STRICTEMENT un JSON conforme au schéma.

Règles strictes :
- Sortie = JSON UNIQUEMENT (aucun commentaire, aucune prose).
- N’utilise que les caractères JSON : {{ }} [ ] , : " .
- Dates au format YYYY-MM-DD ; si inconnu -> null (si jour ou mois inconnu, mets 01 par défaut).
- N'invente rien ; si une information est absente -> null ou [].
- Respecte toutes les clés attendues par le schéma.

Schéma JSON :
{json.dumps(schema, ensure_ascii=False, indent=2)}

Contraintes meta :
- meta.langue = "fr"
- meta.date_extraction = "{today}"
- meta.modele_utilise = "{MODEL_NAME}"
- meta.schema_version = "1.0"

Texte source :
<<<
{text}
>>>

⚠️ IMPORTANT :
- Ta sortie DOIT être un seul objet JSON valide commençant par '{{' et se terminant par '}}'.
- NE PRODUIS AUCUN TEXTE EN DEHORS DU JSON.
"""

# ─────────────────────────────────────────────
# 4. Appel API Mistral
# ─────────────────────────────────────────────
def call_mistral_api(prompt: str) -> str:
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == "REPLACE_ME":
        raise RuntimeError(
            "Clé API Mistral absente. Définis MISTRAL_API_KEY dans l'environnement ou le .env."
        )
    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Tu es un assistant médical spécialisé en extraction structurée. Réponds UNIQUEMENT en JSON valide."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # déterministe
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# 5. Extraction JSON robuste
# ─────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """
    Extrait un objet JSON valide depuis le texte renvoyé par le modèle.
    Stratégies :
    1) bloc ```json ... ```
    2) bloc ``` ... ```
    3) plus grand objet { ... } par comptage d'accolades
    """
    if not text or not text.strip():
        raise ValueError("Réponse vide du modèle.")

    # 1) ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) ``` ... ```
    m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) Trouver le plus grand objet JSON par empilement d'accolades
    s = text
    start_positions = [i for i, c in enumerate(s) if c == "{"]
    if not start_positions:
        raise ValueError("Aucune accolade ouvrante '{' trouvée — pas de JSON détecté.")

    best = None
    for start in start_positions:
        depth = 0
        for j in range(start, len(s)):
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:j+1]
                    try:
                        parsed = json.loads(candidate)
                        if (best is None) or (len(candidate) > len(best)):
                            best = candidate
                        # on continue à chercher pour éventuellement un objet plus grand
                    except json.JSONDecodeError:
                        pass
                    break
    if best:
        return json.loads(best)

    raise ValueError("Impossible d'extraire un JSON valide de la réponse du modèle.")

# ─────────────────────────────────────────────
# 6. Validation JSON vs schéma
# ─────────────────────────────────────────────
def validate_json(data: dict, schema: dict):
    try:
        validate(instance=data, schema=schema)
        print("✅ JSON conforme au schéma !")
    except ValidationError as e:
        print("❌ Erreur de validation :", e.message)
        raise

# ─────────────────────────────────────────────
# 7. Retry/Réparation (optionnel mais utile)
# ─────────────────────────────────────────────
def repair_json_with_model(raw_output: str, schema: dict) -> dict:
    """
    Demande au modèle de reformater en JSON pur si la 1re sortie n'était pas un JSON valide.
    """
    repair_prompt = f"""
Ta sortie précédente n'était pas un JSON pur.
Reformate STRICTEMENT en un seul objet JSON valide conforme au schéma ci-dessous.
NE RAJOUTE AUCUN TEXTE HORS JSON.

Schéma:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Sortie précédente:
<<<
{raw_output}
>>>
"""
    repaired = call_mistral_api(repair_prompt)
    return extract_json(repaired)

# ─────────────────────────────────────────────
# 8. Sauvegardes / logs
# ─────────────────────────────────────────────
def save_run_artifacts(run_dir: Path, text: str, prompt: str, raw: str, data: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "extracted_text.txt").write_text(text, encoding="utf-8")
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "raw_output.txt").write_text(raw, encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ─────────────────────────────────────────────
# 9. Pipeline complet
# ─────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"run_{ts}"

    print("📄 Extraction du texte...")
    text = extract_text_from_pdf(PDF_PATH)
    print("Longueur du texte extrait :", len(text))

    print("\n🧠 Chargement du schéma...")
    schema = json.loads(Path(SCHEMA_PATH).read_text(encoding="utf-8"))

    print("\n🧩 Construction du prompt...")
    prompt = build_prompt(text, schema)

    print("\n🌐 Envoi au modèle Mistral...")
    raw_output = call_mistral_api(prompt)
    print("\n🗒️ Réponse du modèle (aperçu) :\n", raw_output[:800], "...\n")

    # Tentative 1 : extraction et validation
    try:
        data = extract_json(raw_output)
        validate_json(data, schema)
    except Exception as e:
        print(f"⚠️ Première tentative échouée ({e}). Essai de réparation...")
        # Tentative 2 : réparation
        data = repair_json_with_model(raw_output, schema)
        validate_json(data, schema)

    # Sauvegarde finale
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "result_mistral_api.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("💾 Résultat sauvegardé dans outputs/result_mistral_api.json")

    # Logs détaillés de la run
    save_run_artifacts(run_dir, text, prompt, raw_output, data)
    print(f"🗂️ Artefacts enregistrés dans: {run_dir}")

if __name__ == "__main__":
    main()
