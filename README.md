# Personal AI — Hackathon Nebius x NVIDIA 2026

Assistant IA personnel avec mémoire de recherche traçable : un modèle
NVIDIA open source exécuté en local, couplé à Tavily pour la recherche
web, avec un registre de sources et une vérification stricte avant
citation (anti-hallucination).

## Choix techniques et contexte

Ce projet a été adapté après deux blocages indépendants de notre
volonté, documentés ici par transparence :
- Inscription à **Nebius TokenFactory** bloquée (restriction
  géographique, Bénin non supporté).
- Paiement refusé pour un **Nebius Endpoint** (carte bancaire non
  acceptée par le système de facturation Nebius), après avoir
  entièrement configuré un endpoint vLLM avec un modèle NVIDIA
  Nemotron (voir `docs/nebius-endpoint-config.md` si conservé).

Solution retenue : exécution 100% locale du modèle NVIDIA (via
`transformers` + quantification 4-bit), ce qui satisfait l'exigence
"au moins un modèle open source NVIDIA" sans dépendre d'un service
tiers payant.

## Modèle

`nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` — architecture Llama
standard (pas de compilation Mamba nécessaire), chargé en 4-bit,
testé sur GPU T4 16 Go (Google Colab gratuit).

## Fonctionnalités

- **Agent de recherche** (`run_agent`) : boucle recherche → extraction
  → synthèse, avec citations `[Sx]` limitées aux sources réellement
  vérifiées par Tavily Extract (garde-fou anti-hallucination).
- **Démonstration Tavily complète** (`run_full_showcase`) : exploite
  les 5 API Tavily — Search (filtré par domaine/période), Map
  (cartographie de site), Crawl (extraction multi-pages), Extract
  (reranking par pertinence) et Research (synthèse autonome).

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # puis renseigne ta clé TAVILY_API_KEY dans .env
```

⚠️ Nécessite un GPU (testé sur T4 16 Go). En local sans GPU, le
chargement du modèle échouera ou sera extrêmement lent.

## Utilisation

```bash
# Poser une question à l'agent
python personal_ai_hackathon.py agent "Analyse les informations sur NVIDIA Rubin en 2026"

# Lancer la démonstration complète des 5 API Tavily
python personal_ai_hackathon.py showcase

# Tests rapides (modèle + Tavily)
python personal_ai_hackathon.py test
```

Ou directement en Python / notebook :

```python
from personal_ai_hackathon import run_agent, run_full_showcase

print(run_agent("Ta question ici"))
run_full_showcase()
```

## Limites connues

- Le tool calling est géré manuellement (JSON forcé par prompt), le
  modèle tournant en local sans API `tools=[...]` façon OpenAI/
  OpenRouter — moins fiable qu'un tool calling natif, d'où les
  garde-fous de validation ajoutés (format tolérant, anti-hallucination
  d'URL, interdiction de conclure sans source vérifiée).
- `crawl()` peut renvoyer des entrées à 0 caractère pour des fichiers
  non-HTML (PDF, téléchargements) — comportement normal de l'API.
- `research()` peut prendre plusieurs minutes ; utiliser `check_research(request_id)`
  pour reprendre un résultat sans reconsommer de crédits.
