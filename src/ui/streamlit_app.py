# src/ui/streamlit_app.py
import streamlit as st
import requests
import os
from datetime import datetime

# Environment variable for API (used in Docker/K8s)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/ask")

st.set_page_config(
    page_title="Financial Agent RAG",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 🎨 CUSTOM CSS (Premium Look)
# ==============================================================================
st.markdown("""
<style>
    /* Rounded chat bubbles */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Better source cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/briefcase.png", width=60)
    st.title("Financial Agent")
    st.caption("Hybrid RAG (Milvus + Elasticsearch)")
    st.divider()

    top_k = st.slider(
        "Number of sources (Top-K)",
        min_value=3, max_value=10, value=5,
        help="Number of final chunks passed to the LLM. More = more context but slower."
    )

    with st.expander("🎯 Advanced Filters", expanded=False):
        company_filter = st.selectbox(
            "Company", options=["", "AAPL", "MSFT", "TSLA", "GOOGL"],
            format_func=lambda x: "All" if x == "" else x,
        )
        period_filter = st.text_input("Period (e.g., FY2024, Q42025)", placeholder="Leave empty for all")
        doctype_filter = st.selectbox(
            "Document Type", options=["", "10-K", "10-Q", "earnings_transcript", "news"],
            format_func=lambda x: "All" if x == "" else x,
        )

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.rerun()

    st.divider()
    st.caption(f"🔗 API connected: `{API_URL.split('/')[2]}`")

# ==============================================================================
# MAIN
# ==============================================================================
st.title("💼 Financial Agent RAG")
st.markdown("Ask questions about financial results, risks, or strategies of major companies.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d-%H%M%S")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📚 {len(message['sources'])} sources consulted", expanded=False):
                for src in message["sources"]:
                    with st.container(border=True):
                        icon = "📊" if src.get("is_table") else "📄"
                        page_info = f" · Page {src['page']}" if src.get("page") else ""
                        score_info = f" · Score {src['score']:.3f}" if src.get("score") else ""
                        st.markdown(f"**{src['rank']}. {icon} {src['document']}** *({src['document_type']}{page_info}{score_info})*")
                        st.caption(f"ID: `{src['chunk_id']}`")
                        st.info(f"{src['snippet']}...")

# User input
if prompt := st.chat_input("Ex: What was Alphabet's revenue in 2024?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Loading status with steps
        with st.status("🔄 Interrogating RAG pipeline...", expanded=True) as status:
            try:
                st.write("🔍 Vector & Lexical search (Milvus + ES)...")
                payload = {"query": prompt, "top_k": top_k}
                if company_filter: payload["company"] = company_filter
                if period_filter: payload["period"] = period_filter
                if doctype_filter: payload["document_type"] = doctype_filter

                # Increased timeout for CPU processing
                response = requests.post(API_URL, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()

                st.write("🧠 Generating answer with Qwen...")
                status.update(label="✅ Answer generated!", state="complete", expanded=False)
                
            except requests.exceptions.Timeout:
                status.update(label="⏰ Timeout", state="error")
                st.error("Service overloaded (CPU). Try reducing the Top-K.")
                st.stop()
            except requests.exceptions.ConnectionError:
                status.update(label="🔌 API unreachable", state="error")
                st.error("FastAPI backend is down.")
                st.stop()
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Unexpected error: {e}")
                st.stop()

        # Display answer
        answer = data.get("answer", "No answer.")
        sources = data.get("sources", [])
        elapsed = data.get("elapsed_s", 0)

        st.markdown(answer)

        # Copy button hack
        st.code(answer, language="markdown") 

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Sources", len(sources))
        col2.metric("Time", f"{elapsed:.1f} s")
        col3.metric("Requested Top-K", top_k)

        # Feedback (Simulated)
        col_f1, col_f2, _ = st.columns([1, 1, 8])
        if col_f1.button("👍", key=f"up_{len(st.session_state.messages)}"):
            st.toast("Thanks for your feedback!", icon="✅")
        if col_f2.button("👎", key=f"down_{len(st.session_state.messages)}"):
            st.toast("Feedback saved for improvement.", icon="📝")

        if sources:
            with st.expander(f"📚 {len(sources)} sources consulted", expanded=True):
                for src in sources:
                    with st.container(border=True):
                        icon = "📊" if src.get("is_table") else "📄"
                        page_info = f" · Page {src['page']}" if src.get("page") else ""
                        score_info = f" · Score {src['score']:.3f}" if src.get("score") else ""
                        st.markdown(f"**{src['rank']}. {icon} {src['document']}** *({src['document_type']}{page_info}{score_info})*")
                        st.caption(f"ID: `{src['chunk_id']}`")
                        st.info(f"{src['snippet']}...")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

st.divider()
st.caption(f"💬 Session: `{st.session_state.conversation_id}` | {len(st.session_state.messages)} messages")