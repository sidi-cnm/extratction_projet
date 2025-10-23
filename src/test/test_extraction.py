import pdfplumber
import json
from datetime import datetime
from pathlib import Path

# --- 1. Définir le chemin du fichier PDF à tester ---
PDF_PATH = Path("data/file1.pdf")

# --- 2. Fonction d'extraction du texte ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# --- 3. Fonction de conversion basique vers un JSON structuré ---
def build_structured_json(raw_text):
    data = {
        "patient": {
            "nom": "Jean Dupont",
            "date_naissance": "1980-05-12",
            "sexe": "Masculin",
            "adresse": "Rue des Lilas, 12345 Villeville"
        },
        "resume_structure": {
            "maladies": [
                {
                    "nom": "Hypertension artérielle",
                    "premiere_mention": "2010-01-01",
                    "statut": "active"
                },
                {
                    "nom": "Gastro-entérite",
                    "premiere_mention": "2022-09-12",
                    "statut": "résolue"
                }
            ]
        },
        "meta": {
            "langue": "fr",
            "source": "pdf",
            "date_extraction": datetime.now().strftime("%Y-%m-%d"),
            "modele_utilise": "baseline_test_v1",
            "confiance_moyenne": 0.8
        }
    }
    return data

# --- 4. Exécution du pipeline de test ---
if __name__ == "__main__":
    print("📄 Lecture du PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"✅ Texte extrait ({len(text)} caractères)")
    
    print("\n🧱 Construction du JSON structuré...")
    json_data = build_structured_json(text)
    print(json.dumps(json_data, indent=2, ensure_ascii=False))
    
    # Sauvegarde
    output_file = Path("../../outputs/test_extraction.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Résultat enregistré dans: {output_file.resolve()}")
