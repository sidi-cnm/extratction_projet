import torch, json, re
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .settings import settings

today = datetime.now().strftime("%Y-%m-%d")

def build_prompt(raw_text: str) -> str:
    # ⚠️ Pas de [INST] ici : on utilise apply_chat_template qui gère le format de discussion.
    return f"""Tu es un extracteur clinique.
Objectif : À partir du texte source, produis STRICTEMENT un JSON conforme au schéma.

Règles strictes :
- Sortie = JSON UNIQUEMENT (aucun commentaire, aucune prose).
- N’utilise que les caractères JSON : {{ }} [ ] , : " .
- Dates au format YYYY-MM-DD ; si inconnu -> null.
- N'invente rien ; si une information est absente -> null ou [].
- Respecte toutes les clés attendues par le schéma.

Schéma condensé :
- patient (id|null, nom, date_naissance, sexe, adresse|null)
- antecedents_medicaux[] (condition, date_diagnostic|null, status|null, type|null, gravite|null)
- traitements_actuels[] (medicament, dose|null, posologie|null, indication|null, debut_traitement|null, fin_traitement|null)
- consultations[] (date, motif|null, observations|null, diagnostic|null, traitement_prescrit|null)
- examens[] (date, type, resultat|null)
- resume_structure (maladies[] {{nom, premiere_mention|null, statut|null, derniere_consultation|null, confiance|null[0..1]}}, allergies[], traitements[])
- meta (langue, source, date_extraction, modele_utilise, confiance_moyenne|null[0..1], schema_version)
- document_source (nom_fichier, type, id_document|null)

Contraintes meta :
- meta.langue = "fr"
- meta.date_extraction = "{today}"
- meta.modele_utilise = "{settings.model_name}"
- meta.schema_version = "1.0"

Texte source :
<<<
{raw_text}
>>>

NE PRODUIS AUCUN TEXTE EN DEHORS DU JSON.
"""

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}\s*$", text, flags=re.S)
    return m.group(0) if m else ""

class ModelWrapper:
    def __init__(self):
        self.model_name = settings.model_name

        # 🔁 Option B : charger local si disponible, sinon HF Hub
        src = settings.model_local_dir
        if not src or not Path(src).exists():
            src = self.model_name  # fallback

        self.tok = AutoTokenizer.from_pretrained(src)

        quant = None
        if settings.use_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            src,
            device_map="auto",
            quantization_config=quant,
            torch_dtype=torch.float16 if quant else "auto",
            low_cpu_mem_usage=True,
        )

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def generate(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "Tu es un extracteur clinique."},
            {"role": "user", "content": build_prompt(text)},
        ]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                do_sample=False
            )

        # ➜ ne décoder que les nouveaux tokens (on enlève l'input)
        gen_only = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen_only, skip_special_tokens=True)

# singleton (chargement 1 seule fois)
model_wrapped: ModelWrapper | None = None
def get_model() -> ModelWrapper:
    global model_wrapped
    if model_wrapped is None:
        model_wrapped = ModelWrapper()
    return model_wrapped
