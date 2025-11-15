import torch, json, re
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ---------- Configuration modèle ----------
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"   # 3.8 B → ~2 Go

# Si tu veux essayer le 8-bit : load_in_8bit=True
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

print("🚀 Chargement du modèle...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu"
)


if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---------- Construction du prompt ----------
today = datetime.now().strftime("%Y-%m-%d")

def build_prompt(raw_text: str) -> str:
    return f"""[INST]
Tu es un extracteur clinique.
Objectif : À partir du texte source, produis STRICTEMENT un JSON conforme au schéma.

Règles:
- Sortie = JSON UNIQUEMENT (aucun commentaire).
- Dates au format YYYY-MM-DD ; si inconnu -> null.
- N'invente rien ; si absent -> null ou [].
- Respecte les clés attendues par le schéma.

Schéma condensé:
- patient (id|null, nom, date_naissance, sexe, adresse|null)
- antecedents_medicaux[] (condition, date_diagnostic|null, status|null, type|null, gravite|null)
- traitements_actuels[] (medicament, dose|null, posologie|null, indication|null, debut_traitement|null, fin_traitement|null)
- consultations[] (date, motif|null, observations|null, diagnostic|null, traitement_prescrit|null)
- examens[] (date, type, resultat|null)
- resume_structure (maladies[] {{nom, premiere_mention|null, statut|null, derniere_consultation|null, confiance|null[0..1]}}, allergies[], traitements[])
- meta (langue, source, date_extraction, modele_utilise, confiance_moyenne|null[0..1], schema_version)
- document_source (nom_fichier, type, id_document|null)

Contraintes meta:
- meta.langue = "fr"
- meta.date_extraction = "{today}"
- meta.modele_utilise = "Mistral-7B-Instruct"
- meta.schema_version = "1.0"

Texte source :
<<<
{raw_text}
>>>

Réponds par le JSON uniquement.
[/INST]"""

# ---------- Génération ----------
def generate_json_from_text(raw_text: str, max_new_tokens=1200):
    messages = [
        {"role": "system", "content": "Tu es un extracteur clinique."},
        {"role": "user", "content": build_prompt(raw_text)},
    ]

    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
        )

    # ➜ On retire les tokens du prompt (pour ne garder que la sortie)
    gen_only = out_ids[0][inputs["input_ids"].shape[1]:]
    output = tok.decode(gen_only, skip_special_tokens=True)
    return output

# ---------- Exécution principale ----------
if __name__ == "__main__":
    raw_text = open("data/file1.pdf", "r", encoding="utf-8").read()
    print("🧾 Taille du texte extrait :", len(raw_text))

    output = generate_json_from_text(raw_text)
    print("✅ Sortie générée :\n", output[:800], "\n...\n")

    # Sauvegarde du résultat brut
    with open("outputs/mistral_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("💾 Résultat sauvegardé dans outputs/mistral_result.json")
