# src/rag/agent.py
"""
Agent RAG financier.
"""
import logging
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.config.settings import settings
from src.rag.search_engine import SearchEngine
from src.rag.query_analyzer import detect_query_type, decompose_query

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==================== INITIALISATION ====================
search_engine = SearchEngine()

logger.info(
    f"🤖 LLM Ollama : {settings.llm_model} @ {settings.ollama_base_url}"
)
llm = ChatOllama(
    model=settings.llm_model,
    temperature=0.0,
    base_url=settings.ollama_base_url,
)

# ==================== FORMATAGE ====================
def format_docs(docs: List[Dict], start_idx: int = 0, max_chars_per_doc: int = 1200) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        company = doc.get("company", "Unknown")
        period = doc.get("period", "Unknown")
        doc_type = doc.get("document_type", "Unknown")
        text = doc.get("text", "")

        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "... [truncated]"

        formatted.append(
            f"[Source {start_idx + i + 1}] Company: {company} | "
            f"Period: {period} | Type: {doc_type}\n"
            f"Excerpt:\n{text}\n"
        )
    return "\n---\n".join(formatted)

# ==================== PROMPTS (100% ENGLISH) ====================
SIMPLE_PROMPT = """
You are an expert financial analyst. Answer the question based ONLY on the provided documents.

**GENERAL RULES:**
1. If the information exists (even in English or in a table), provide it.
2. Synonyms: "revenue" = "total revenues" = "net sales", "net income" = "profit".
3. Carefully read the tables (rows AND columns).
4. If nothing is found: "I could not find this information in the provided documents."
5. **LANGUAGE RULE (STRICT)**: You MUST answer in **ENGLISH**, regardless of the language of the question or the documents.

**🚨 ANTI-ERROR RULE: QUARTER vs YEAR (CRITICAL)**
In a "Q4 2025 Results" press release, numbers in the first paragraph are ALWAYS for the QUARTER, even if the title mentions "Fiscal Year".
Example: "Consolidated revenues increased 18% to 113.8 billion" -> 113.8 B is **Q4 2025 only**. Do NOT present it as FY2025.

**How to find the true FY:**
- PRIORITY 1: Look for a chunk with "Year Ended December 31" + 3 columns (e.g., 2023, 2024, 2025). The last number is the FY.
- PRIORITY 2: Look for "Full year", "Twelve months ended", "Annual revenues".
- PRIORITY 3: Sum the 4 quarters if all are available.
- Otherwise: say "not available".

**UNITS (CRUCIAL):**
- 10-K/10-Q tables: amounts are in **millions USD** ("in millions").
  -> "$402,836" = 402,836 million = **402.8 billion**.
- Always write both forms: "402,836 M USD (402.8 B USD)".
- Reminder: 1K = 1,000; 1M = 1,000,000; 1B = 1 billion.

**PERIOD DISTINCTION:**
- "FY2025" / "in 2025" = full fiscal year.
- "Q1 2025" / "Q4 2025" = specific quarter.
- If the question asks for the year, answer with the FY (not Q4).

Context documents:
{context}

Question: {question}

Answer (in English ONLY):
"""

COMPARATIVE_PROMPT = """
You are an expert financial analyst. Follow these rules STRICTLY.

**RULE 1 — ONE SOURCE = ONE COMPANY**
Each [Source X] relates to ONE specific company indicated in the "Company:" field.
You MUST NEVER use a figure from a Microsoft source for Alphabet (and vice versa).

**RULE 2 — SUMMARY TABLE**
Fill this table with the figures FOUND in the sources:

| Company | Period | TOTAL Revenue | Source |
|---|---|---|---|
| Alphabet | 2024 | ?? | [Source X] |
| Microsoft | 2024 | ?? | [Source Y] |

- For each line, cite the EXACT [Source X] from which the figure comes.
- If you cannot find it, write "❌ not found" and DO NOT GUESS.

**RULE 3 — CROSS-VERIFICATION**
Before finalizing, verify that:
- The Alphabet figure comes from a source with "Company: GOOGL/Alphabet".
- The Microsoft figure comes from a source with "Company: MSFT/Microsoft".
- If the 2 figures are identical, it is SUSPICIOUS -> double-check the sources.

**RULE 4 — ALLOWED EXCERPTS**
Look primarily for:
- "Total revenues", "Revenues" in a "Year Ended December 31" table.
- "Total revenue" in an "INCOME STATEMENT".
- Take the column corresponding to the requested year (not the previous ones).

**RULE 5 — LANGUAGE**
You MUST answer in **ENGLISH**.

Context documents:
{context}

Question: {question}

Answer (in English ONLY):
"""

RISK_PROMPT = """
You are an expert financial analyst. You must extract and synthesize the **risk factors** from the provided documents.

**Rules:**
1. List risks by category if possible (macroeconomic, regulatory, operational, competitive).
2. Provide a brief description (1-2 sentences) and its source [Source X] for each risk.
3. Use bullet points for readability.
4. Do not fabricate any risk that is not explicitly mentioned.
5. **LANGUAGE RULE**: You MUST answer in **ENGLISH**.

Context documents:
{context}

Question: {question}

Identified risk factors (in English ONLY):
"""

# ==================== AGENT PRINCIPAL ====================
def ask_financial_agent(
    question: str,
    top_k: int = 5,
    filters: Optional[Dict] = None,
    force_simple: bool = False,
):
    """
    Returns (answer: str, sources: List[Dict]).
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
            return "No relevant documents found.", []

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
        return "No relevant documents found.", []

    # Construction du contexte groupé
    context_parts = []
    source_cursor = 0
    for i, item in enumerate(all_docs_by_query):
        header = f"### Sub-question {i + 1}: {item['sub_query']}"
        if item.get("company"):
            header += f" (Company: {item['company']})"
        context_parts.append(header)
        context_parts.append(
            format_docs(item["docs"], start_idx=source_cursor)
        )
        source_cursor += len(item["docs"])
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