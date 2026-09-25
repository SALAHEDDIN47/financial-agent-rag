# src/ui/streamlit_app.py
import streamlit as st
import requests
import os
from datetime import datetime

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/ask")

st.set_page_config(
    page_title="Financial Agent RAG",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("💼 Financial Agent")
    st.caption("RAG hybride Milvus + Elasticsearch")

    st.divider()

    # ✅ FIX : top_k EST envoyé à l'API
    top_k = st.slider(
        "Nombre de sources",
        min_value=3,
        max_value=10,
        value=5,
        help="Nombre de chunks finaux passés au LLM. Plus = plus de contexte mais plus lent."
    )

    # ✅ NOUVEAU : filtres optionnels
    with st.expander("🎯 Filtres avancés", expanded=False):
        company_filter = st.selectbox(
            "Entreprise",
            options=["", "AAPL", "MSFT", "TSLA", "GOOGL"],
            format_func=lambda x: "Toutes" if x == "" else x,
        )
        period_filter = st.text_input(
            "Période (ex: FY2024, Q42025)",
            value="",
            placeholder="Laisser vide pour toutes",
        )
        doctype_filter = st.selectbox(
            "Type de document",
            options=["", "10-K", "10-Q", "earnings_transcript", "news"],
            format_func=lambda x: "Tous" if x == "" else x,
        )

    st.divider()

    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.rerun()

    st.divider()
    st.caption(f"🔗 API : {API_URL}")


# ==============================================================================
# MAIN
# ==============================================================================
st.title("💼 Financial Agent RAG")
st.markdown(
    "Posez vos questions sur les résultats financiers, les risques "
    "ou les stratégies des entreprises (Tesla, Apple, Microsoft, Alphabet)."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d-%H%M%S")


# Historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(
                f"📚 {len(message['sources'])} sources consultées", expanded=False
            ):
                for src in message["sources"]:
                    icon = "📊" if src.get("is_table") else "📄"
                    page_info = f" · p.{src['page']}" if src.get("page") else ""
                    score_info = f" · score {src['score']:.3f}" if src.get("score") else ""
                    st.markdown(
                        f"**{src['rank']}. {icon} {src['document']}**"
                        f" *({src['document_type']}{page_info}{score_info})*"
                    )
                    st.caption(f"`{src['chunk_id']}`")
                    st.markdown(f"> {src['snippet']}...")
                    st.divider()


# Input utilisateur
if prompt := st.chat_input("Ex: Quel était le chiffre d'affaires d'Alphabet en 2024 ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche hybride + reranking..."):
            try:
                payload = {"query": prompt, "top_k": top_k}
                if company_filter:
                    payload["company"] = company_filter
                if period_filter:
                    payload["period"] = period_filter
                if doctype_filter:
                    payload["document_type"] = doctype_filter

                response = requests.post(API_URL, json=payload, timeout=280)
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "Pas de réponse.")
                sources = data.get("sources", [])
                elapsed = data.get("elapsed_s", 0)

                st.markdown(answer)

                # ✅ Indicateurs de qualité
                col1, col2, col3 = st.columns(3)
                col1.metric("Sources", len(sources))
                col2.metric("Temps", f"{elapsed:.1f} s")
                col3.metric("Top-K demandé", top_k)

                if sources:
                    with st.expander(
                        f"📚 {len(sources)} sources consultées", expanded=True
                    ):
                        for src in sources:
                            icon = "📊" if src.get("is_table") else "📄"
                            page_info = f" · p.{src['page']}" if src.get("page") else ""
                            score_info = (
                                f" · score {src['score']:.3f}" if src.get("score") else ""
                            )
                            st.markdown(
                                f"**{src['rank']}. {icon} {src['document']}**"
                                f" *({src['document_type']}{page_info}{score_info})*"
                            )
                            st.caption(f"`{src['chunk_id']}`")
                            st.markdown(f"> {src['snippet']}...")
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except requests.exceptions.Timeout:
                st.error("⏰ Timeout (>180s). Le service est peut-être surchargé.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 API injoignable. Lancez `uv run uvicorn src.api.main:app`.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Erreur HTTP : {e}")
            except Exception as e:
                st.error(f"❌ Erreur inattendue : {e}")

st.divider()
st.caption(
    f"💬 Conversation : `{st.session_state.conversation_id}` "
    f"| {len(st.session_state.messages)} messages"
)