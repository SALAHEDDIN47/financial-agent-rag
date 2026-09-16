# src/rag/query_analyzer.py
import logging
import re
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


# ==================== DÉTECTION HEURISTIQUE ====================
COMPARATIVE_KEYWORDS = [
    "compare", "comparer", "comparaison", "versus", "vs", "différence",
    "plus que", "moins que", "mieux que", "écart", "évolution entre",
    "par rapport à", "contre",
]

RISK_KEYWORDS = [
    "risque", "risques", "risk", "risks", "facteurs de risque",
    "danger", "menace", "incertitude", "vulnérabilité",
]

MULTI_COMPANY_PATTERN = re.compile(
    r"\b(Tesla|Apple|Microsoft|Alphabet|Google|Amazon|Meta|Netflix|GOOGL|MSFT|TSLA|AAPL)\b",
    re.IGNORECASE,
)


def detect_query_type(query: str) -> Dict[str, any]:
    """
    Analyse la question et retourne un dictionnaire décrivant son type :
    - is_comparative : booléen
    - is_risk_query : booléen
    - companies : liste d'entreprises détectées
    - needs_decomposition : booléen (si la question est complexe)
    """
    query_lower = query.lower()

    # Détection comparative
    is_comparative = any(kw in query_lower for kw in COMPARATIVE_KEYWORDS)

    # Détection "risques"
    is_risk_query = any(kw in query_lower for kw in RISK_KEYWORDS)

    # Détection des entreprises
    companies_found = list(set(
        m.group(0).upper() for m in MULTI_COMPANY_PATTERN.finditer(query)
    ))
    # Normaliser Google → GOOGL, etc.
    companies = []
    for c in companies_found:
        if c in ["GOOGLE", "ALPHABET"]:
            companies.append("GOOGL")
        elif c in ["TESLA"]:
            companies.append("TSLA")
        elif c in ["MICROSOFT"]:
            companies.append("MSFT")
        elif c in ["APPLE"]:
            companies.append("AAPL")
        else:
            companies.append(c)

    # Décision de décomposition
    needs_decomposition = is_comparative or is_risk_query or len(companies) >= 2

    return {
        "is_comparative": is_comparative,
        "is_risk_query": is_risk_query,
        "companies": list(set(companies)),
        "needs_decomposition": needs_decomposition,
    }


# ==================== DÉCOMPOSITION VIA LLM ====================
DECOMPOSITION_PROMPT = """You are a financial analysis expert.
Decompose a complex question into simple, independent sub-questions, IN ENGLISH.

**Rules:**
1. ALL sub-questions must be in ENGLISH (to optimize search).
2. If the question compares companies, create one sub-question per company.
3. Each sub-question must be standalone.
4. Return ONLY a JSON list, no surrounding text.

**Examples:**

Question: "Compare Tesla and Microsoft revenue in 2024"
Response:
[
  {{"sub_query": "What was Tesla's revenue in 2024?", "company": "TSLA", "aspect": "revenue"}},
  {{"sub_query": "What was Microsoft's revenue in 2024?", "company": "MSFT", "aspect": "revenue"}}
]

**Now decompose:**
Question: "{question}"

Response (JSON only):
"""


def decompose_query(query: str, llm: ChatOllama) -> List[Dict]:
    """
    Utilise le LLM pour décomposer une question complexe en sous-questions.
    """
    import json

    prompt = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT)
    chain = prompt | llm

    try:
        response = chain.invoke({"question": query})
        # Extraction du texte
        content = response.content if hasattr(response, 'content') else str(response)
        if isinstance(content, list):
            content = "".join(
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in content
            )

        # Nettoyage du JSON (au cas où le LLM ajoute du texte)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        sub_queries = json.loads(content)
        logger.info(f"✅ Question décomposée en {len(sub_queries)} sous-questions")
        return sub_queries

    except Exception as e:
        logger.warning(f"⚠️ Échec de la décomposition : {e}")
        # Fallback : on retourne la question originale
        return [{"sub_query": query, "company": None, "aspect": "general"}]