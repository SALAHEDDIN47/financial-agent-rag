# src/ui/streamlit_app.py
import streamlit as st
import requests
import os
import json
from datetime import datetime

# URL de l'API FastAPI
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/ask")

st.set_page_config(
    page_title="Financial Agent RAG",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/1200px-Microsoft_logo.svg.png", width=80)
    st.title("⚙️ Paramètres")
    
    # Sélection du nombre de documents à récupérer
    top_k = st.slider("Nombre de documents à consulter", min_value=1, max_value=10, value=5, help="Plus le nombre est élevé, plus la réponse est précise mais plus elle est lente.")
    
    # Bouton pour réinitialiser la conversation
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    Cet agent utilise une recherche hybride (vectorielle + lexicale) pour répondre à vos questions financières.
    
    **Données disponibles** : Rapports 10-K et rapports annuels d'entreprises (Tesla, Apple, Microsoft, Alphabet, etc.)
    """)
    
    st.divider()
    st.caption(f"🔗 API : {API_URL}")

# --- Titre principal ---
st.title("💼 Financial Agent RAG")
st.markdown("Posez vos questions sur les résultats financiers, les risques ou les stratégies des entreprises.")

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d-%H%M%S")

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources consultées", expanded=False):
                cols = st.columns(min(len(message["sources"]), 2))
                for idx, source in enumerate(message["sources"]):
                    col = cols[idx % 2]
                    with col:
                        st.markdown(f"**📄 {source.get('document', 'Document')}**")
                        st.caption(f"🔖 {source.get('chunk_info', '')}")
                        st.markdown(f"> *{source.get('snippet', '')[:200]}...*")
                        st.divider()

# Zone de saisie utilisateur
if prompt := st.chat_input("Ex: Quel était le chiffre d'affaires d'Alphabet en 2024 ?"):
    # Ajouter le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Appel à l'API
    with st.chat_message("assistant"):
        with st.spinner("🔍 L'agent analyse les documents financiers..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": prompt, "top_k": 5},
                    timeout=180
                )
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("answer", "Désolé, je n'ai pas pu générer de réponse.")
                sources = data.get("sources", [])
                
                # Afficher la réponse
                st.markdown(answer)
                
                # Afficher les sources
                if sources:
                    with st.expander("📚 Sources consultées", expanded=True):
                        for i, source in enumerate(sources):
                            doc = source.get('document', 'Document inconnu')
                            info = source.get('chunk_info', '')
                            snippet = source.get('snippet', '')
                            st.markdown(f"**{i+1}. {doc}** ({info})")
                            st.markdown(f"> *{snippet}*")
                            st.divider()
                
                # Ajouter au session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except requests.exceptions.Timeout:
                st.error("⏰ La requête a expiré. L'agent met trop de temps à répondre.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Impossible de se connecter à l'API. Vérifiez que le serveur FastAPI est lancé.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Erreur : {e}")
            except Exception as e:
                st.error(f"❌ Une erreur inattendue est survenue : {e}")

# Pied de page
st.divider()
st.caption(f"💬 Conversation ID : {st.session_state.conversation_id} | {len(st.session_state.messages)} messages échangés")