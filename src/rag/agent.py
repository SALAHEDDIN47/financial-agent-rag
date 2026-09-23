# src/rag/agent.py
import logging
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.rag.search_engine import SearchEngine
from src.rag.query_analyzer import detect_query_type, decompose_query

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# ==================== INITIALISATION ====================
search_engine = SearchEngine(
    milvus_host="localhost",
    milvus_port="19530",
    es_host="http://localhost:9200",
)

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    base_url="http://localhost:11434",
)


# ==================== FORMATAGE ====================
def format_docs(docs: List[Dict], max_chars_per_doc: int = 1200) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        company = doc.get("company", "Unknown")
        period = doc.get("period", "Unknown")
        doc_type = doc.get("document_type", "Unknown")
        text = doc.get("text", "")

        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "... [tronqué]"

        formatted.append(
            f"[Source {i + 1}] Entreprise: {company} | "
            f"Période: {period} | Type: {doc_type}\n"
            f"Extrait:\n{text}\n"
        )
    return "\n---\n".join(formatted)


# ==================== PROMPTS ====================
SIMPLE_PROMPT = """
Tu es un analyste financier expert. Réponds à la question en te basant UNIQUEMENT sur les documents fournis.

**Règles :**
1. Si l'information existe (même en anglais ou dans un tableau), donne-la.
2. Synonymes : "chiffre d'affaires" = "revenue", "bénéfice" = "net income".
3. Lis attentivement les tableaux (lignes ET colonnes).
4. Si rien n'est trouvé : "Je n'ai pas trouvé cette information dans les documents fournis."
5. **LANGUE** : Réponds dans la même langue que la question.
6. Cite tes sources au format [Source X].

**UNITÉS (CRUCIAL)** :
- Tableaux 10-K/10-Q : montants en **millions USD** ("in millions")
  → "$17,693" = 17,693 millions = **17.7 milliards de dollars**
- Les revenus des grandes entreprises (Apple, Tesla, Google, Microsoft) se lisent en **milliards**
- Écris toujours les 2 formes : "17,693 M USD (17.7 Md USD)"
- Rappel : 1 000 = 1K ; 1 000 000 = 1M ; 1 000 000 000 = 1B = 1 milliard

**DISTINCTION DES PÉRIODES (CRUCIAL)** :
- "FY2025" / "in 2025" = exercice fiscal complet
- "Q1 2025" / "Q4 2025" = trimestre spécifique
- Si la question demande un trimestre ET que tu as le trimestre → **réponds directement**, sans mentionner de différence FY vs Q
- Si la question demande l'année ET que tu n'as que le trimestre → signale la différence

Documents de contexte :
{context}

Question : {question}

Réponse :
"""

COMPARATIVE_PROMPT = """
Tu es un analyste financier expert. Suis EXACTEMENT ces étapes.

**ÉTAPE 1 — LECTURE** : Parcours chaque [Source X] et note mentalement :
   - Son entreprise (champ "Entreprise:")
   - Sa période (champ "Période:")
   - S'il contient un chiffre de TOTAL revenue/revenue (pas une variation)

**ÉTAPE 2 — EXTRACTION** : Pour chaque entreprise demandée dans la question :
   - Cherche la source dont la Période correspond à la question
   - Extrais le chiffre TOTAL (pas la variation)
   - Exemple de TOTAL : "revenues ... to $90.2 billion"
   - Exemple de VARIATION (à IGNORER) : "revenue increased 12%"

**ÉTAPE 3 — RÉPONSE** :

### Microsoft
- **Revenue [Période]** : [chiffre OU "❌ Non disponible"]

### Alphabet
- **Revenue [Période]** : [chiffre OU "❌ Non disponible"]

### Comparison
[si 2 chiffres disponibles]

**RÈGLES ABSOLUES :**
- Ne JAMAIS inventer.
- Ne JAMAIS multiplier un trimestriel par 4.
- Ne JAMAIS substituer Q4 2025 à Q1 2025.
- Cite [Source X] après chaque chiffre.
- Réponds dans la langue de la question.

Documents de contexte :
{context}

Question : {question}

Réponse :
"""

RISK_PROMPT = """
Tu es un analyste financier expert. Tu dois extraire et synthétiser les **facteurs de risque** à partir des documents fournis.

**Règles :**
1. Liste les risques par catégorie si possible (macroéconomique, réglementaire, opérationnel, concurrentiel).
2. Donne pour chaque risque une brève description (1-2 phrases) et sa source [Source X].
3. Utilise des puces pour la lisibilité.
4. Ne fabrique aucun risque qui n'est pas explicitement mentionné.

Documents de contexte :
{context}

Question : {question}

Facteurs de risque identifiés :
"""


# ==================== AGENT PRINCIPAL ====================
def ask_financial_agent(
    question: str,
    top_k: int = 5,
    filters: Optional[Dict] = None,
    force_simple: bool = False,
):
    """
    Retourne (answer: str, sources: List[Dict]).
    """
    logger.info(f"🔍 Question reçue : '{question}'")

    analysis = (
        detect_query_type(question)
        if not force_simple
        else {"needs_decomposition": False}
    )

    # --- CAS SIMPLE ---
    if not analysis.get("needs_decomposition", False):
        docs = search_engine.search_hybrid(question, top_k=top_k, filters=filters)
        if not docs:
            return "Aucun document pertinent trouvé.", []

        context = format_docs(docs)
        prompt = ChatPromptTemplate.from_template(SIMPLE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return answer, docs

    # --- CAS COMPLEXE ---
    sub_queries = decompose_query(question, llm)

    all_docs_by_query = []
    all_docs_flat = []
    for sq in sub_queries:
        sub_q = sq.get("sub_query", question)
        sub_filters = dict(filters) if filters else {}
        if sq.get("company"):
            sub_filters["company"] = sq["company"]

        docs = search_engine.search_hybrid(
            sub_q, top_k=top_k, filters=sub_filters or None
        )
        all_docs_by_query.append(
            {"sub_query": sub_q, "company": sq.get("company"), "docs": docs}
        )
        all_docs_flat.extend(docs)

    # Déduplication
    seen = set()
    unique_docs = []
    for doc in all_docs_flat:
        cid = doc.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            unique_docs.append(doc)

    if not unique_docs:
        return "Aucun document pertinent trouvé.", []

    # Construction du contexte groupé
    context_parts = []
    for i, item in enumerate(all_docs_by_query):
        header = f"### Sous-question {i + 1} : {item['sub_query']}"
        if item.get("company"):
            header += f" (Entreprise : {item['company']})"
        context_parts.append(header)
        context_parts.append(format_docs(item["docs"]))
    context = "\n\n".join(context_parts)

    # Choix du prompt
    if analysis.get("is_risk_query"):
        prompt = ChatPromptTemplate.from_template(RISK_PROMPT)
    elif analysis.get("is_comparative") or len(analysis.get("companies", [])) >= 2:
        prompt = ChatPromptTemplate.from_template(COMPARATIVE_PROMPT)
    else:
        prompt = ChatPromptTemplate.from_template(SIMPLE_PROMPT)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, unique_docs


# ==================== TEST ====================
if __name__ == "__main__":
    queries = [
        "What was Tesla Q4 2025 revenue?",
        "Compare Alphabet and Microsoft revenue in Q1 2025",
        "Quels sont les principaux facteurs de risque mentionnés dans le 10-K de Tesla 2025 ?",
    ]

    for q in queries:
        print("\n" + "=" * 70)
        print(f"❓ {q}")
        print("=" * 70)
        answer, sources = ask_financial_agent(q, top_k=5)
        print(answer)
        print(f"\n[SOURCES: {len(sources)}]")
        print()