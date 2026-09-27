#!/usr/bin/env python3
# ============================================================
# PERSONAL AI HACKATHON 2026 — Nebius x NVIDIA
#
# Assistant IA personnel avec mémoire de recherche traçable :
# - Modèle NVIDIA open source, exécuté localement (4-bit) :
#   nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1
# - Recherche web via Tavily (Search, Extract, Map, Crawl, Research)
# - Tool calling géré manuellement (JSON forcé par prompt), car le
#   modèle tourne en local sans API "tools=[...]" façon OpenAI
# - Registre de sources avec scoring qualité/pertinence et
#   vérification stricte avant citation (anti-hallucination)
#
# Ce fichier fusionne :
#   - l'agent principal (recherche + synthèse avec citations)
#   - le module de démonstration Tavily (5 API : search, map,
#     crawl, extract, research)
#
# ⚠️ Nécessite un GPU (testé sur T4 16 Go, via Google Colab).
#    Pas de dépendance stricte à Colab : la clé API se lit dans
#    une variable d'environnement TAVILY_API_KEY (voir .env.example),
#    avec repli automatique sur les secrets Colab si disponibles.
#
# Installation : pip install -r requirements.txt
# Utilisation  : voir README.md
# ============================================================

import json
import os
import re
import time

from urllib.parse import urlparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tavily import TavilyClient


# ============================================================
# 0. SECRETS — portable Colab / local / CI
# ============================================================
# Priorité : variable d'environnement classique (.env, export
# shell, secret CI). Si absente ET qu'on tourne dans Colab, on
# retombe sur les secrets Colab pour ne pas casser le workflow
# de développement initial.
# ============================================================

def get_secret(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value

    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name)
    except Exception:
        return ""


TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError(
        "❌ TAVILY_API_KEY introuvable. Définis-la comme variable "
        "d'environnement (voir .env.example) ou comme secret Colab."
    )

print("✅ TAVILY_API_KEY chargée")


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

MAX_ITERATIONS = 6
MAX_TOTAL_TOOL_CALLS = 12
MAX_TOOL_CALLS_PER_TURN = 3  # le parsing JSON manuel est plus fragile ; on limite
MAX_SEARCH_RESULTS = 5
MAX_EXTRACT_URLS = 2
SEARCH_CONTENT_LIMIT = 600
EXTRACT_CONTENT_LIMIT = 1200

# Ce qu'on renvoie réellement au modèle après un appel d'outil
# (indépendant des limites ci-dessus, qui servent au registre de
# sources / à l'affichage) — c'est CE texte qui rentre dans le
# contexte du modèle et fait grossir la mémoire GPU à chaque tour.
TOOL_RESULT_TO_MODEL_LIMIT = 1500

MAX_NEW_TOKENS = 512

# Nombre de tours (paires assistant/outil) conservés dans
# l'historique envoyé au modèle. Au-delà, on ne garde que les
# plus récents pour éviter que le contexte n'explose en mémoire.
MAX_HISTORY_TURNS_KEPT = 3

DEBUG_RAW_ARGUMENTS = True

# Sujet et domaines officiels par défaut pour la démo Tavily
# (module 2 de ce fichier) — modifiable selon ta requête.
SUJET_DEMO_TAVILY = "NVIDIA Rubin"
DOMAINES_OFFICIELS = [
    "nvidia.com",
    "nvidianews.nvidia.com",
    "developer.nvidia.com",
    "investor.nvidia.com",
    "blogs.nvidia.com",
]


# ============================================================
# 2. CHARGEMENT DU MODÈLE (quantifié 4-bit) + CLIENT TAVILY
# ============================================================

print("⏳ Chargement du modèle (peut prendre quelques minutes)...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print("✅ Modèle chargé :", MODEL_NAME)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
print("✅ Client Tavily initialisé")


# ============================================================
# 3. NORMALISATION DES URLS
# ============================================================

def normalize_url(value):
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = text.replace("\\/", "/").replace("\\", "")

    matches = re.findall(r"https?://[^\s<>\[\]\"']+", text)
    if not matches:
        return ""

    result = matches[-1]
    result = result.rstrip(".,;:!?'\")]}>`").strip()

    if not result.startswith(("http://", "https://")):
        return ""

    if "[" in result or "](" in result:
        return ""

    return result


# ============================================================
# 4. DOMAINES (scoring qualité des sources)
# ============================================================

PRIMARY_DOMAINS = {
    "nvidia.com": 100,
    "nvidianews.nvidia.com": 100,
    "developer.nvidia.com": 100,
    "investor.nvidia.com": 100,
    "blogs.nvidia.com": 100,
}

TECHNICAL_PRIMARY_DOMAINS = {
    "micron.com": 90,
    "investors.micron.com": 90,
    "samsung.com": 90,
    "news.samsung.com": 90,
    "skhynix.com": 90,
    "news.skhynix.com": 90,
    "tsmc.com": 90,
}

SECONDARY_DOMAINS = {
    "reuters.com": 80,
    "anandtech.com": 80,
    "techpowerup.com": 80,
    "tomshardware.com": 80,
    "computerbase.de": 75,
}


def get_domain(url):
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def source_quality_score(url):
    domain = get_domain(url)
    if domain in PRIMARY_DOMAINS:
        return PRIMARY_DOMAINS[domain]
    if domain in TECHNICAL_PRIMARY_DOMAINS:
        return TECHNICAL_PRIMARY_DOMAINS[domain]
    if domain in SECONDARY_DOMAINS:
        return SECONDARY_DOMAINS[domain]
    return 40


def relevance_score(query, title, url):
    query_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower()))
    text = f"{title} {url}".lower()

    score = 0
    for word in query_words:
        if word in text:
            score += 5

    important_terms = [
        "rubin", "vera", "nvidia", "gpu", "hbm4", "nvlink",
        "architecture", "2026", "ai", "cpx", "nvl72", "blackwell",
    ]
    for term in important_terms:
        if term in text:
            score += 8

    return score


# ============================================================
# 5. REGISTRE DES SOURCES
# ============================================================

used_sources = []
verified_sources = set()


def register_source(title, url, source_type="search", query=""):
    url = normalize_url(url)
    if not url:
        return None

    for source in used_sources:
        if source["url"] == url:
            if title and source["title"] == "Page extraite":
                source["title"] = title
            if source_type == "extract":
                source["type"] = "extract"
                source["extracted"] = True
                source["verified"] = True
                verified_sources.add(source["id"])
            return source

    quality = source_quality_score(url)
    relevance = relevance_score(query, title or "", url)

    source = {
        "id": f"S{len(used_sources) + 1}",
        "title": title or "Source sans titre",
        "url": url,
        "type": source_type,
        "quality": quality,
        "relevance": relevance,
        "total_score": quality + relevance,
        "extracted": source_type == "extract",
        "verified": source_type == "extract",
    }

    used_sources.append(source)
    if source["verified"]:
        verified_sources.add(source["id"])

    return source


# ============================================================
# 6. TAVILY SEARCH + EXTRACT (utilisés par l'agent)
# ============================================================

def tavily_search_tool(query: str, search_depth: str = "advanced"):
    query = str(query).strip()
    if not query:
        return {
            "error": "La requête Tavily est vide.",
            "answer": "",
            "sources": [],
            "recommended_extract_urls": [],
        }

    try:
        print("\n🔎 Tavily Search :", query)

        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=MAX_SEARCH_RESULTS,
            include_answer=True,
            include_raw_content=False,
        )

        answer = response.get("answer", "") or ""
        sources = []

        for result in response.get("results", []):
            title = result.get("title", "Source sans titre") or "Source sans titre"
            url = normalize_url(result.get("url", ""))
            content = result.get("content", "") or ""

            if not url:
                continue

            source = register_source(title=title, url=url, source_type="search", query=query)
            if not source:
                continue

            sources.append({
                "id": source["id"],
                "title": source["title"],
                "url": source["url"],
                "quality": source["quality"],
                "relevance": source["relevance"],
                "total_score": source["total_score"],
                "verified": source["verified"],
                "content": content[:SEARCH_CONTENT_LIMIT],
            })

        sources.sort(key=lambda item: item["total_score"], reverse=True)

        high_quality_sources = [s for s in sources if s["quality"] >= 75]
        selected_sources = high_quality_sources if high_quality_sources else sources

        recommended_extract_urls = []
        for source in selected_sources:
            clean_url = normalize_url(source["url"])
            if clean_url and clean_url not in recommended_extract_urls:
                recommended_extract_urls.append(clean_url)
            if len(recommended_extract_urls) >= MAX_EXTRACT_URLS:
                break

        print("\n📊 Classement des sources :")
        for source in sources:
            print(f"{source['id']} | score={source['total_score']} | "
                  f"qualité={source['quality']} | {source['title']} | {source['url']}")

        print("\n⭐ Sources candidates pour Extract :")
        for url in recommended_extract_urls:
            print("-", url)

        return {
            "answer": answer,
            "sources": sources,
            "recommended_extract_urls": recommended_extract_urls,
        }

    except Exception as e:
        return {
            "error": "Erreur Tavily Search : " + str(e),
            "answer": "",
            "sources": [],
            "recommended_extract_urls": [],
        }


def tavily_extract_tool(urls: list):
    if not isinstance(urls, list):
        return {"error": "urls doit être une liste.", "results": []}

    cleaned_urls = []
    for raw_url in urls:
        clean = normalize_url(raw_url)
        if clean and clean not in cleaned_urls:
            cleaned_urls.append(clean)

    cleaned_urls = cleaned_urls[:MAX_EXTRACT_URLS]

    for url in cleaned_urls:
        if not url.startswith(("http://", "https://")):
            raise ValueError("❌ URL invalide avant Tavily Extract : " + str(url))
        if "[" in url or "](" in url:
            raise ValueError("❌ URL Markdown détectée avant Tavily Extract : " + str(url))

    print("\n🧹 URLs réellement envoyées à Tavily Extract :")
    for url in cleaned_urls:
        print("-", url)

    if not cleaned_urls:
        return {"error": "Aucune URL valide.", "results": []}

    try:
        response = tavily_client.extract(urls=cleaned_urls)
        results = []

        for index, result in enumerate(response.get("results", [])):
            returned_url = normalize_url(result.get("url", ""))

            if returned_url and returned_url in cleaned_urls:
                source_url = returned_url
            elif index < len(cleaned_urls):
                source_url = cleaned_urls[index]
            else:
                source_url = returned_url

            if not source_url:
                continue

            registered_source = None
            for source in used_sources:
                if source["url"] == source_url:
                    registered_source = source
                    break

            if not registered_source:
                title = result.get("title", "") or "Page extraite"
                registered_source = register_source(title=title, url=source_url, source_type="extract")

            raw_content = (result.get("raw_content", "") or "")[:EXTRACT_CONTENT_LIMIT]
            is_verified = bool(raw_content.strip())

            if is_verified:
                registered_source["verified"] = True
                registered_source["extracted"] = True
                registered_source["type"] = "extract"
                verified_sources.add(registered_source["id"])

            results.append({
                "id": registered_source["id"],
                "title": registered_source["title"],
                "url": source_url,
                "verified": is_verified,
                "raw_content": raw_content,
            })

        print("📖 Pages réellement extraites :", len(results))
        for result in results:
            status = "✅" if result["verified"] else "⚠️"
            print("   ", status, result["id"], result["url"])

        verified_registry = [
            {"id": s["id"], "title": s["title"], "url": s["url"]}
            for s in used_sources if s["verified"]
        ]

        return {"results": results, "verified_sources": verified_registry}

    except Exception as e:
        return {"error": str(e), "results": [], "verified_sources": []}


# ============================================================
# 7. SYSTEM PROMPT — TOOL CALLING MANUEL
# ============================================================

TOOLS_DESCRIPTION = """
Tu as accès à deux outils :

1. tavily_search_tool(query: string, search_depth: "basic" ou "advanced")
   → Recherche des informations actuelles sur Internet.

2. tavily_extract_tool(urls: liste de chaînes)
   → Lit en profondeur une ou plusieurs pages web trouvées pendant une recherche.
   → Utilise en priorité les URLs recommandées par la dernière recherche.
"""

SYSTEM_PROMPT = f"""detailed thinking off

Tu es un assistant de recherche IA personnel.

{TOOLS_DESCRIPTION}

============================================================
RÈGLE ABSOLUE DE FORMAT
============================================================
Tu dois répondre UNIQUEMENT avec un objet JSON valide, rien d'autre
avant ou après (pas de texte, pas de balises markdown ```json).

Deux formats possibles, et RIEN d'autre :

Pour appeler un outil :
{{"action": "tool_call", "tool": "<nom_outil>", "arguments": {{...}}}}

Pour donner la réponse finale (uniquement quand tu as assez
d'informations VÉRIFIÉES) :
{{"action": "final_answer", "content": "<ta synthèse avec citations [S1] [S2] ...>"}}

============================================================
STRATÉGIE DE RECHERCHE
============================================================
- Utilise tavily_search_tool pour des informations récentes ou vérifiables.
- Privilégie les sources officielles et primaires.
- Utilise ensuite tavily_extract_tool sur les sources les plus pertinentes
  pour lire leur contenu en profondeur.
- Une source ne devient VÉRIFIÉE que si tavily_extract_tool a réellement
  renvoyé du contenu pour elle.
- N'utilise QUE les IDs de la liste CURRENT VERIFIED SOURCES pour les
  citations [Sx] dans ta réponse finale. N'invente jamais d'ID ni d'URL.
- Arrête la recherche dès que tu as assez de preuves vérifiées.

============================================================
EXEMPLE OBLIGATOIRE À SUIVRE
============================================================
Pour la toute première réponse à une question, tu dois TOUJOURS
commencer par une recherche, jamais par une final_answer. Exemple
de première réponse correcte pour la question
"Analyse les informations sur NVIDIA Rubin" :

{{"action": "tool_call", "tool": "tavily_search_tool", "arguments": {{"query": "NVIDIA Rubin 2026", "search_depth": "advanced"}}}}

N'écris RIEN d'autre que l'objet JSON. Pas d'explication, pas de
texte avant ou après, pas de balises ```.
"""


def build_dynamic_system_prompt():
    verified_registry = [
        {"id": s["id"], "title": s["title"], "url": s["url"]}
        for s in used_sources if s["verified"]
    ]

    registry_text = json.dumps(verified_registry, ensure_ascii=False, indent=2)

    return (
        SYSTEM_PROMPT
        + "\n\n============================================================\n"
        + "CURRENT VERIFIED SOURCES\n"
        + "============================================================\n\n"
        + registry_text
        + "\n\nONLY THESE SOURCE IDS MAY BE CITED AS VERIFIED:\n"
        + ", ".join(sorted(verified_sources))
    )


# ============================================================
# 8. GÉNÉRATION LOCALE
# ============================================================

def local_chat(messages):
    """
    Envoie `messages` (format [{"role": ..., "content": ...}, ...])
    au modèle chargé localement et renvoie le texte généré (str).
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    del inputs, output_ids, generated
    torch.cuda.empty_cache()

    return text.strip()


# ============================================================
# 9. PARSING JSON DE LA RÉPONSE DU MODÈLE
# ============================================================

def extract_json_object(text):
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None

    return None


def extract_urls_from_text(text):
    if not text:
        return []

    matches = re.findall(r"https?://[^\s<>\[\]\"']+", str(text))
    cleaned = []
    for url in matches:
        clean = normalize_url(url)
        if clean and clean not in cleaned:
            cleaned.append(clean)
    return cleaned


# ============================================================
# 10. PARSE DES ARGUMENTS D'OUTIL
# ============================================================

def parse_tool_arguments(function_name, raw_arguments, fallback_urls=None):
    if fallback_urls is None:
        fallback_urls = []

    if DEBUG_RAW_ARGUMENTS:
        print("🔍 RAW arguments reçus pour", function_name, ":", repr(raw_arguments))

    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}

    if function_name == "tavily_extract_tool":
        raw_urls = arguments.get("urls", [])
        if isinstance(raw_urls, str):
            raw_urls = [raw_urls]
        if not isinstance(raw_urls, list):
            raw_urls = []

        model_urls = []
        for url in raw_urls:
            cleaned = normalize_url(url)
            if cleaned and cleaned not in model_urls:
                model_urls.append(cleaned)

        cleaned_fallback = []
        for url in fallback_urls:
            cleaned = normalize_url(url)
            if cleaned and cleaned not in cleaned_fallback:
                cleaned_fallback.append(cleaned)

        # Sécurité anti-hallucination : on n'accepte du modèle que
        # les URLs identiques à une URL recommandée par la dernière
        # recherche. Toute URL déformée est ignorée.
        trusted_urls = [url for url in model_urls if url in cleaned_fallback]

        if trusted_urls:
            return {"urls": trusted_urls[:MAX_EXTRACT_URLS]}

        return {"urls": cleaned_fallback[:MAX_EXTRACT_URLS]}

    if function_name == "tavily_search_tool":
        query = arguments.get("query", "")
        if not isinstance(query, str):
            query = str(query)
        query = query.strip()

        depth = arguments.get("search_depth", "advanced")
        if depth not in ("basic", "advanced"):
            depth = "advanced"

        return {"query": query, "search_depth": depth}

    return arguments


# ============================================================
# 11. EXECUTION DES OUTILS
# ============================================================

def execute_tool(function_name, arguments):
    if function_name == "tavily_search_tool":
        if not arguments.get("query"):
            return {"error": "Requête de recherche vide.", "sources": [], "answer": ""}
        return tavily_search_tool(**arguments)

    if function_name == "tavily_extract_tool":
        return tavily_extract_tool(**arguments)

    return {"error": "Outil inconnu : " + str(function_name)}


# ============================================================
# 12. SOURCES FINALES + VALIDATION DES CITATIONS
# ============================================================

def build_sources_section():
    verified = [s for s in used_sources if s["verified"]]

    if not verified:
        return "\n\n### Sources vérifiées\nAucune source n'a été vérifiée par Tavily Extract."

    verified.sort(key=lambda s: (s["quality"], s["relevance"]), reverse=True)

    text = "\n\n### Sources vérifiées par Tavily Extract\n"
    for source in verified[:10]:
        text += f"- [{source['id']}] {source['title']}\n  {source['url']}\n"

    return text


def validate_citations(text):
    if not text:
        return

    cited_ids = set(re.findall(r"\[(S\d+)\]", text))
    invalid_ids = [sid for sid in cited_ids if sid not in verified_sources]

    if invalid_ids:
        print("\n⚠️ ATTENTION : citations non vérifiées :", sorted(invalid_ids))
    else:
        print("\n✅ Toutes les citations [Sx] de la réponse correspondent à des sources vérifiées.")


# ============================================================
# 13. RUN AGENT (module 1 : assistant personnel search+extract)
# ============================================================

def run_agent(user_prompt: str, max_iterations=MAX_ITERATIONS):
    used_sources.clear()
    verified_sources.clear()

    last_search_urls = []
    total_tool_calls = 0

    messages = [
        {"role": "system", "content": build_dynamic_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    print("\n🧠 Question :", user_prompt)
    print("=" * 70)

    for iteration in range(1, max_iterations + 1):
        print(f"\n🔄 Tour agent : {iteration}")

        messages[0]["content"] = build_dynamic_system_prompt()

        raw_text = local_chat(messages)
        print("\n🗣️ Sortie brute du modèle :\n", raw_text[:800])

        parsed = extract_json_object(raw_text)

        if parsed is not None and parsed.get("action") not in ("tool_call", "final_answer"):
            loose_action = str(parsed.get("action", "")).lower()
            if "extract" in loose_action:
                parsed = {"action": "tool_call", "tool": "tavily_extract_tool",
                          "arguments": parsed.get("arguments", {})}
            elif "search" in loose_action:
                parsed = {"action": "tool_call", "tool": "tavily_search_tool",
                          "arguments": parsed.get("arguments", {})}

        if parsed is None or "action" not in parsed:
            print("⚠️ JSON invalide ou incomplet — on redemande le format correct.")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    "Ta réponse n'était pas un JSON valide au format demandé. "
                    "Réponds UNIQUEMENT avec un objet JSON "
                    '{"action": "tool_call", ...} ou {"action": "final_answer", ...}.'
                ),
            })
            continue

        if parsed.get("action") == "final_answer":
            if not verified_sources:
                print("\n🚫 Le modèle a tenté de conclure sans aucune source "
                      "vérifiée — refusé, on le renvoie vers une recherche.")

                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Tu ne peux PAS donner de final_answer : aucune source "
                        "n'est encore vérifiée. Tu dois d'abord appeler "
                        "tavily_search_tool, puis tavily_extract_tool sur les "
                        "résultats pertinents. Réponds maintenant avec un "
                        '{"action": "tool_call", "tool": "tavily_search_tool", '
                        '"arguments": {...}}.'
                    ),
                })
                continue

            final_answer = str(parsed.get("content", "")).strip()
            validate_citations(final_answer)
            return final_answer + build_sources_section()

        if parsed.get("action") == "tool_call":
            function_name = parsed.get("tool", "")
            raw_arguments = parsed.get("arguments", {})

            arguments = parse_tool_arguments(function_name, raw_arguments, fallback_urls=last_search_urls)

            print("\n🔧 Outil :", function_name)
            print("📋 Arguments :", arguments)

            if total_tool_calls >= MAX_TOTAL_TOOL_CALLS:
                result = {"error": "Budget de recherche atteint."}
            else:
                result = execute_tool(function_name, arguments)
                total_tool_calls += 1

            if function_name == "tavily_search_tool" and isinstance(result, dict):
                recommended_urls = result.get("recommended_extract_urls", [])
                last_search_urls = []
                for url in recommended_urls:
                    clean = normalize_url(url)
                    if clean and clean not in last_search_urls:
                        last_search_urls.append(clean)
                    if len(last_search_urls) >= MAX_EXTRACT_URLS:
                        break

            elif function_name == "tavily_extract_tool" and isinstance(result, dict):
                if verified_sources:
                    print("🔐 Sources actuellement vérifiées :", sorted(verified_sources))

            tool_result_message = ("Résultat de l'outil :\n"
                                    + json.dumps(result, ensure_ascii=False)[:TOOL_RESULT_TO_MODEL_LIMIT])

            if function_name == "tavily_search_tool" and last_search_urls:
                tool_result_message += (
                    "\n\nTon PROCHAIN message doit être obligatoirement :\n"
                    '{"action": "tool_call", "tool": "tavily_extract_tool", '
                    '"arguments": {"urls": ' + json.dumps(last_search_urls, ensure_ascii=False) + '}}'
                )

            messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            messages.append({"role": "user", "content": tool_result_message})

            max_messages = 1 + (MAX_HISTORY_TURNS_KEPT * 2)
            if len(messages) > max_messages:
                messages = [messages[0]] + messages[-(max_messages - 1):]

            continue

        print("⚠️ Action inconnue dans la réponse du modèle :", parsed.get("action"))
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({
            "role": "user",
            "content": 'Action non reconnue. Utilise "tool_call" ou "final_answer".',
        })

    print("\n⚠️ Limite d'itérations atteinte.")

    messages[0]["content"] = build_dynamic_system_prompt()
    messages.append({
        "role": "user",
        "content": (
            'Produis maintenant {"action": "final_answer", "content": "..."} '
            "en utilisant uniquement les informations vérifiées et les IDs "
            "présents dans CURRENT VERIFIED SOURCES."
        ),
    })

    raw_text = local_chat(messages)
    parsed = extract_json_object(raw_text) or {}
    final_answer = str(parsed.get("content", raw_text)).strip()

    validate_citations(final_answer)
    return final_answer + build_sources_section()


# ============================================================
# 14. DÉMONSTRATION TAVILY (module 2 : les 5 API Tavily)
# ============================================================
# Déterministe (pas piloté par le LLM) — fiabilité maximale pour
# une démo live. Montre une utilisation large et réfléchie de
# Tavily : search filtré, map, crawl, extract avec reranking,
# et research (synthèse autonome de bout en bout).
# ============================================================

def _section(titre):
    print("\n" + "=" * 70)
    print(titre)
    print("=" * 70)


def demo_search(sujet=SUJET_DEMO_TAVILY):
    _section("1. SEARCH — recherche avancée filtrée par domaine et période")

    response = tavily_client.search(
        query=f"{sujet} announcement specifications",
        search_depth="advanced",
        topic="news",
        time_range="year",
        include_domains=DOMAINES_OFFICIELS,
        max_results=8,
        include_answer=True,
    )

    print("Réponse résumée par Tavily :", response.get("answer", "")[:300])
    print(f"\n{len(response.get('results', []))} résultats (restreints aux domaines officiels) :")

    for r in response.get("results", []):
        print(f"- [{r.get('score', 0):.2f}] {r.get('title')}\n  {r.get('url')}")

    return response


def demo_map(sujet=SUJET_DEMO_TAVILY):
    _section("2. MAP — cartographie du site NVIDIA News")

    response = tavily_client.map(
        url="https://nvidianews.nvidia.com",
        max_depth=2,
        limit=30,
        instructions=f"Trouve les pages liées à {sujet}",
    )

    urls_trouvees = []
    for item in response.get("results", []):
        url = item.get("url") if isinstance(item, dict) else item
        if url:
            urls_trouvees.append(url)

    print(f"{len(urls_trouvees)} URLs découvertes sur le site :")
    for url in urls_trouvees[:15]:
        print("-", url)

    return urls_trouvees


def demo_crawl(sujet=SUJET_DEMO_TAVILY):
    _section("3. CRAWL — extraction multi-pages guidée par instructions")

    response = tavily_client.crawl(
        url="https://nvidianews.nvidia.com",
        max_depth=2,
        max_breadth=20,
        limit=10,
        instructions=f"Trouve les annonces et communiqués de presse sur {sujet}",
        extract_depth="advanced",
        chunks_per_source=3,
    )

    pages = response.get("results", [])
    print(f"{len(pages)} pages extraites via crawl :")

    for page in pages:
        content_len = len(page.get("raw_content", "") or "")
        print(f"- {page.get('url')} ({content_len} caractères)")

    return pages


def demo_extract(urls, sujet=SUJET_DEMO_TAVILY):
    _section("4. EXTRACT — extraction ciblée avec reranking par pertinence")

    if not urls:
        print("Aucune URL fournie, extraction ignorée.")
        return {}

    response = tavily_client.extract(
        urls=urls[:3],
        extract_depth="advanced",
        query=f"caractéristiques techniques et date de sortie de {sujet}",
        chunks_per_source=3,
    )

    for r in response.get("results", []):
        print(f"\n--- {r.get('url')} ---")
        print((r.get("raw_content", "") or "")[:400])

    return response


def demo_research(sujet=SUJET_DEMO_TAVILY, max_wait_seconds=480):
    """
    ⚠️ Peut prendre plusieurs minutes (parfois 5-10 min) — c'est le
    fonctionnement normal de l'API pour une synthèse de qualité.
    """
    _section("5. RESEARCH — recherche et synthèse autonomes par Tavily")

    result = tavily_client.research(
        input=f"Analyse les spécifications techniques, la date de disponibilité "
              f"et l'écosystème de partenaires de la plateforme {sujet} en 2026.",
        model="auto",
    )

    request_id = result["request_id"]
    print("Requête de recherche lancée, id :", request_id)
    print("(Peut prendre plusieurs minutes — c'est normal pour ce type de recherche.)")

    return check_research(request_id, max_wait_seconds=max_wait_seconds)


def check_research(request_id, max_wait_seconds=480):
    """
    Permet de revenir vérifier plus tard une recherche déjà lancée,
    sans relancer un nouvel appel research() (donc sans reconsommer
    de crédits).
    """
    response = tavily_client.get_research(request_id)
    waited = 0

    while response.get("status") not in ("completed", "failed") and waited < max_wait_seconds:
        time.sleep(15)
        waited += 15
        response = tavily_client.get_research(request_id)
        print(f"... statut après {waited}s : {response.get('status')}")

    if response.get("status") == "completed":
        print("\n📄 Rapport généré par Tavily Research :\n")
        print(response.get("content", "")[:2000])
    elif response.get("status") == "failed":
        print("❌ La recherche a échoué côté Tavily.")
    else:
        print(f"⏳ Toujours en cours après {waited}s (statut : {response.get('status')}).\n"
              f"Repasse plus tard avec : check_research('{request_id}')")

    return response


def run_full_showcase(sujet=SUJET_DEMO_TAVILY):
    search_response = demo_search(sujet)
    mapped_urls = demo_map(sujet)
    crawled_pages = demo_crawl(sujet)

    top_search_urls = [
        r["url"] for r in search_response.get("results", [])
        if r.get("score", 0) > 0.3
    ]
    demo_extract(top_search_urls, sujet)

    demo_research(sujet, max_wait_seconds=90)

    _section("✅ DÉMONSTRATION TERMINÉE — 5/5 API Tavily exploitées")
    print("1. search()   — filtré par domaines officiels + période")
    print("2. map()      — structure du site découverte")
    print(f"   → {len(mapped_urls)} URLs cartographiées")
    print("3. crawl()    — pages extraites en profondeur")
    print(f"   → {len(crawled_pages)} pages")
    print("4. extract()  — extraction ciblée avec reranking")
    print("5. research() — synthèse autonome de bout en bout")


# ============================================================
# 15. TESTS RAPIDES
# ============================================================

def test_model():
    text = local_chat([{"role": "user", "content": "Réponds exactement : connexion réussie."}])
    print("🤖 Modèle :", text)


def test_tavily():
    result = tavily_search_tool("NVIDIA Rubin 2026")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def chat_loop():
    """
    Boucle interactive : tape une question, l'agent répond, et ainsi
    de suite jusqu'à ce que tu tapes "exit" ou "quit". Pratique pour
    tester à la volée ou pour une démo vidéo (poser plusieurs
    questions sans relancer le script/éditer le code).
    """
    print("\n💬 Mode interactif — tape ta question, ou 'exit' pour quitter.\n")

    while True:
        try:
            question = input("Toi : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin de la session.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Fin de la session.")
            break

        reponse = run_agent(question)
        print("\n🤖 Assistant :\n" + reponse + "\n")


# ============================================================
# 16. POINT D'ENTRÉE CLI
# ============================================================
#
# Usage :
#   python personal_ai_hackathon.py agent "Ta question ici"
#   python personal_ai_hackathon.py showcase
#   python personal_ai_hackathon.py test
# ============================================================

if __name__ == "__main__":
    import sys

    print()
    print("=" * 70)
    print("✅ PERSONAL AI HACKATHON 2026 — CHARGÉ ET PRÊT")
    print("=" * 70)
    print("✅ Modèle NVIDIA local (4-bit) :", MODEL_NAME)
    print("✅ Tavily Search + Extract + Map + Crawl + Research")
    print("✅ Aucune dépendance à TokenFactory, Nebius Endpoint ou OpenRouter")

    args = sys.argv[1:]

    if not args:
        print("\nUsage :")
        print('  python personal_ai_hackathon.py agent "Ta question ici"')
        print("  python personal_ai_hackathon.py chat")
        print("  python personal_ai_hackathon.py showcase")
        print("  python personal_ai_hackathon.py test")
    elif args[0] == "agent":
        question = " ".join(args[1:]) or "Analyse les informations disponibles en 2026 sur NVIDIA Rubin."
        print(run_agent(question))
    elif args[0] == "chat":
        chat_loop()
    elif args[0] == "showcase":
        run_full_showcase()
    elif args[0] == "test":
        test_model()
        test_tavily()
    else:
        print(f"Commande inconnue : {args[0]}")
