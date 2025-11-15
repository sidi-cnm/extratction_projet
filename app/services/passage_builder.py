# Transforme le JSON structuré en petits textes indexables + métadonnées
from typing import Dict, List, Tuple

def json_to_passages(mr: Dict) -> List[Tuple[str, Dict]]:
    out: List[Tuple[str, Dict]] = []

    patient = mr.get("patient", {}) or {}
    nom = patient.get("nom") or "Inconnu"
    out.append((
        f"Patient: {nom}. Sexe: {patient.get('sexe')}. Naissance: {patient.get('date_naissance')}.",
        {"section": "patient", "patient_nom": nom}
    ))

    for i, ant in enumerate(mr.get("antecedents_medicaux", []) or []):
        txt = f"Antécédent: {ant.get('condition')}; date diagnostic: {ant.get('date_diagnostic')}."
        out.append((txt, {"section": "antecedent", "idx": i, "condition": ant.get("condition")}))

    for i, mal in enumerate((mr.get("resume_structure") or {}).get("maladies", []) or []):
        txt = (f"Maladie: {mal.get('nom')}. Première mention: {mal.get('premiere_mention')}."
               f" Statut: {mal.get('statut')}. Dernière consultation: {mal.get('derniere_consultation')}.")
        out.append((txt, {"section": "maladie", "idx": i, "nom": mal.get("nom")}))

    for i, tr in enumerate(mr.get("traitements_actuels", []) or []):
        txt = (f"Traitement: {tr.get('medicament')}; dose: {tr.get('dose')}; "
               f"posologie: {tr.get('posologie')}; indication: {tr.get('indication')}.")
        out.append((txt, {"section": "traitement", "idx": i, "medicament": tr.get("medicament")}))

    for i, cons in enumerate(mr.get("consultations", []) or []):
        txt = (f"Consultation du {cons.get('date')}: motif {cons.get('motif')}; "
               f"diagnostic {cons.get('diagnostic')}; traitement {cons.get('traitement_prescrit')}.")
        out.append((txt, {"section": "consultation", "idx": i, "date": cons.get("date")}))
    return out
