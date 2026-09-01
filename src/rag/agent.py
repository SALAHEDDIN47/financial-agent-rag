# src/rag/agent.py
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI  
from src.rag.search_engine import SearchEngine

# Charger les variables d'environnement depuis .env
load_dotenv()

# Vérifier la clé API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ La variable GEMINI_API_KEY n'est pas définie dans le fichier .env")

# Initialisation du moteur de recherche
search_engine = SearchEngine(
    milvus_host="localhost",
    milvus_port="19530",
    es_host="http://localhost:9200"
)

# ✅ CORRECTION DU NOM DU MODÈLE
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",  # Modèle valide et rapide
    temperature=0.0,
    google_api_key=GEMINI_API_KEY
)

def format_docs(docs: List[Dict]) -> str:
    """Formate les documents récupérés pour le prompt du LLM."""
    formatted = []
    for i, doc in enumerate(docs):
        company = doc.get("company", "Unknown")
        period = doc.get("period", "Unknown")
        doc_type = doc.get("document_type", "Unknown")
        text = doc.get("text", "")
        formatted.append(
            f"[Source {i+1}] Entreprise: {company} | Période: {period} | Type: {doc_type}\n"
            f"Contenu: {text}\n"
        )
    return "\n---\n".join(formatted)

# Prompt système robuste pour éviter les hallucinations
prompt_template = """
Tu es un analyste financier expert et un assistant RAG de haute précision.
Ta tâche est de répondre à la question de l'utilisateur en te basant UNIQUEMENT sur les documents fournis ci-dessous.

Règles strictes :
1. Si la réponse ne se trouve pas dans les documents, réponds : "Je n'ai pas trouvé cette information dans les documents fournis."
2. Cite toujours tes sources en utilisant le format [Source X].
3. Sois concis, professionnel et va droit au but.
4. Si la question porte sur des chiffres, donne les chiffres exacts tels qu'ils apparaissent dans le texte.

Documents de contexte :
{context}

Question de l'utilisateur : {question}

Réponse :
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Chaîne RAG (Retrieval-Augmented Generation)
rag_chain = (
    {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_financial_agent(question: str, top_k: int = 5):
    """Fonction principale pour interroger l'agent."""
    print(f"🔍 Recherche d'informations pour : '{question}'...\n")
    
    # 1. Récupération des documents (Hybride)
    docs = search_engine.search_hybrid(question, top_k=top_k)
    
    if not docs:
        return "Aucun document pertinent trouvé."
    
    # 2. Génération de la réponse par le LLM
    print("🤖 Génération de la réponse par l'IA...\n")
    response = rag_chain.invoke({"question": question, "docs": docs})
    
    return response

# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    query = "What was Tesla's GAAP operating income in 2024?"
    answer = ask_financial_agent(query, top_k=5)
    print("="*60)
    print("RÉPONSE DE L'AGENT :")
    print("="*60)
    print(answer)