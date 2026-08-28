import streamlit as st
import os
from pathlib import Path

CONFIG = {
    # Embeddings
    "embedding_model": "intfloat/multilingual-e5-small",
    "embedding_device": "cpu",

    # Découpage
    "chunk_size": 512,
    "chunk_overlap": 100,
    "chunk_separators": ["\n\n", "\n", "Article", "§", ".", " ", ""],
    "chunks_pickle_path": "./chunks.pkl",

    # Recherche
    "hybrid_search_weights": (0.5, 0.5),
    "top_k_initial": 20,
    "top_k_final": 5,

    # LLM
    "llm_model": "gemini-3.0-flash",
    "llm_temperature": 0.2,
    "llm_streaming": True,

    # Chemins
    "docs_directory": "./documents/",
    "chroma_persist_directory": "./chroma_db/",
    "collection_name": "togolese_law_collection",

    # Agentic
    "enable_agentic_retrieval": False,
    "max_retrieval_iterations": 3,
}

from src.document_loader import load_documents, chunk_documents
from src.vector_store import create_vector_store, load_vector_store, load_chunks
from src.retrievers import (
    create_hybrid_retriever,
    agentic_retrieve,
    hybrid_retrieve,
    get_retriever_from_vector_store,
)
from src.llm_client import create_llm
from src.prompts import create_generation_prompt
from src.config import CONFIG


def initialize_pipeline():
    """Initialise le pipeline en chargeant l'index existant si possible."""
    print("\n" + "=" * 60)
    print("🚀 INITIALIZING RAG PIPELINE")
    print("=" * 60)

    llm = create_llm()
    prompt = create_generation_prompt()

    vector_store = load_vector_store()
    hybrid_retriever = None
    is_indexed = False

    if vector_store:
        chunks = load_chunks()

        if chunks:
            print("✅ Chunks trouvés, création du retriever hybride...")
            hybrid_retriever = create_hybrid_retriever(
                chunks,
                vector_store
            )
        else:
            print(
                "⚠️ Chunks introuvables, utilisation du retriever "
                "vectoriel uniquement."
            )
            hybrid_retriever = get_retriever_from_vector_store(
                vector_store
            )

        is_indexed = True
        print("✅ Index chargé avec succès.")

    else:
        print(
            "⚠️ Aucun index existant trouvé. "
            "Veuillez indexer des documents."
        )

    return {
        "llm": llm,
        "prompt": prompt,
        "hybrid_retriever": hybrid_retriever,
        "vector_store": vector_store,
        "is_indexed": is_indexed,
    }


def index_documents(pipeline_state):
    """Indexe les documents du dossier documents/ et met à jour le pipeline."""
    print("\n" + "=" * 60)
    print("🔄 STARTING INDEXATION")
    print("=" * 60)

    docs = load_documents()

    if not docs:
        raise ValueError("Aucun document chargé.")

    chunks = chunk_documents(docs)

    vector_store = create_vector_store(chunks)

    hybrid_retriever = create_hybrid_retriever(
        chunks,
        vector_store
    )

    pipeline_state["vector_store"] = vector_store
    pipeline_state["hybrid_retriever"] = hybrid_retriever
    pipeline_state["is_indexed"] = True

    print("\n✅ Indexation terminée!")
    print(f"   - {len(docs)} documents")
    print(f"   - {len(chunks)} chunks")
    print("=" * 60)

    return pipeline_state


def query_pipeline(pipeline_state, question):
    """Interroge le pipeline et retourne (réponse, documents)."""

    print("\n" + "=" * 60)
    print(f"❓ Question: {question}")
    print("=" * 60)

    llm = pipeline_state["llm"]
    prompt = pipeline_state["prompt"]

    legal_keywords = [
        "loi",
        "article",
        "code",
        "juridique",
        "droit",
        "justice",
        "contrat",
        "travail",
        "licenciement",
        "salarié",
        "employeur",
        "divorce",
        "mariage",
        "succession",
        "héritage",
        "propriété",
        "terrain",
        "foncier",
        "pénal",
        "infraction",
        "crime",
        "délit",
        "tribunal",
        "avocat",
        "plainte",
        "procès",
        "responsabilité",
        "obligation",
        "créance",
        "dette",
        "société",
        "ohada",
        "commerce",
        "civil",
        "famille",
        "togo",
        "togolais",
        "constitution",
        "jurisprudence",
        "peine",
        "amende",
        "prison",
        "bail",
        "location",
        "travailleur",
        "emploi",
        "salaire",
    ]

    question_lower = question.lower()

    is_legal = any(
        keyword in question_lower
        for keyword in legal_keywords
    )

    if not is_legal:
        print("💬 Question non juridique → réponse directe")

        response = llm.invoke(
            prompt.format_messages(
                context="",
                question=question,
                chat_history=[],
            )
        )

        answer = response.content

        if isinstance(answer, list):
            answer = "".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict)
                and block.get("type") == "text"
            )

        return answer, []

    if not pipeline_state["is_indexed"]:
        raise ValueError(
            "La base juridique n'est pas indexée. "
            "Veuillez indexer les documents."
        )

    hybrid_retriever = pipeline_state["hybrid_retriever"]

    if CONFIG["enable_agentic_retrieval"]:
        documents, context_sufficient = agentic_retrieve(
            llm,
            hybrid_retriever,
            question,
        )

        print(
            f"📊 Context sufficient: {context_sufficient}"
        )

    else:
        documents = hybrid_retrieve(
            hybrid_retriever,
            question,
        )

        print(
            f"📊 Retrieved {len(documents)} documents"
        )

    context = "\n\n---\n\n".join(
        [
            (
                f"[Source: {doc.metadata.get('filename', 'Inconnu')}]\n"
                f"{doc.page_content}"
            )
            for doc in documents
        ]
    )

    print("\n🤖 Generating answer...")

    response = (
        prompt | llm
    ).invoke({
        "context": context,
        "question": question,
        "chat_history": [],
    })

    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    print("\n" + "=" * 60)
    print("✅ Answer generated")
    print("=" * 60)

    return answer, documents


def simple_query(pipeline_state, question):
    """Version simplifiée : ne retourne que la réponse."""
    answer, _ = query_pipeline(
        pipeline_state,
        question
    )
    return answer


# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="LexMind • AI Juridique Togolais",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# STYLES CSS
# ---------------------------------------------------------

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-gradient: linear-gradient(135deg, #0A1128 0%, #101F42 50%, #001F54 100%);
    --accent-sky: #38BDF8;
    --accent-sky-glow: rgba(56, 189, 248, 0.3);
    --navy-card: rgba(15, 23, 42, 0.75);
    --navy-border: rgba(56, 189, 248, 0.2);
    --text-white: #F8FAFC;
    --text-muted: #94A3B8;
}

.stApp {
    background: var(--bg-gradient);
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-white);
}

h1, h2, h3, .brand-title {
    font-family: 'Cinzel', serif !important;
    letter-spacing: 0.5px;
}

.hero-header {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--navy-border);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, var(--accent-sky-glow) 0%, transparent 60%);
    pointer-events: none;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #FFFFFF 30%, var(--accent-sky) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: var(--accent-sky);
    font-weight: 500;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.hero-desc {
    color: var(--text-muted);
    font-size: 0.95rem;
    max-width: 650px;
    margin: 0;
}

[data-testid="stSidebar"] {
    background-color: rgba(10, 17, 40, 0.9) !important;
    border-right: 1px solid var(--navy-border);
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-active {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.status-inactive {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(251, 191, 36, 0.3);
}

.stChatInputContainer {
    padding-bottom: 1rem;
}

.stChatInput input {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid var(--navy-border) !important;
    color: white !important;
    border-radius: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stChatInput input:focus {
    border-color: var(--accent-sky) !important;
    box-shadow: 0 0 12px var(--accent-sky-glow) !important;
}

[data-testid="stChatMessage"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    margin-bottom: 1rem !important;
    backdrop-filter: blur(8px);
}

[data-testid="stChatMessage"]:nth-child(even) {
    background-color: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
}

.stButton>button {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5) !important;
}

.streamlit-expanderHeader {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border-radius: 8px !important;
    color: var(--accent-sky) !important;
}
</style>
"""

st.markdown(
    custom_css,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# INITIALISATION DU PIPELINE
# ---------------------------------------------------------

@st.cache_resource
def get_pipeline():
    return initialize_pipeline()


pipeline = get_pipeline()


# ---------------------------------------------------------
# BARRE LATÉRALE
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        "<h2 style='font-size: 1.6rem; margin-bottom: 0.5rem;'>⚖️ LEXMIND</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='color: var(--text-muted); font-size: 0.85rem;'>"
        "Plateforme IA d'intelligence juridique"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    is_indexed = pipeline.get(
        "is_indexed",
        False
    )

    status_class = (
        "status-active"
        if is_indexed
        else "status-inactive"
    )

    status_text = (
        "● Base Indexée & Prête"
        if is_indexed
        else "▲ Indexation Requise"
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.4rem;">
                Statut du système
            </div>
            <div class="status-badge {status_class}">
                {status_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Administration")

    if st.button(
        "🔄 Recharger/Indexer la Base",
        use_container_width=True,
    ):
        with st.spinner(
            "Indexation des textes de loi togolais..."
        ):
            pipeline = index_documents(pipeline)
            st.success(
                "Base de connaissances mise à jour !"
            )
            st.rerun()

    st.markdown("---")

    st.markdown("### 📜 Domaines Couverts")

    st.markdown(
        """
        - **Code Civil & Droit de la Famille**
        - **Code du Travail**
        - **Droit des Affaires & OHADA**
        - **Code Pénal Togolais**
        """
    )

    st.markdown("---")

    st.caption(
        "République Togolaise • Justice - Travail - Patrie"
    )


# ---------------------------------------------------------
# BANNIÈRE PRINCIPALE
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">LEXMIND</div>
        <div class="hero-subtitle">
            <span>⚖️</span>
            Assistant d'Intelligence Juridique Togolais
        </div>
        <p class="hero-desc">
            Explorez la jurisprudence, les codes officiels et les textes
            fondamentaux de la République Togolaise avec une précision
            algorithmique de pointe.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CHAT ET INTERACTION
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:

    avatar = (
        "👤"
        if msg["role"] == "user"
        else "⚖️"
    )

    with st.chat_message(
        msg["role"],
        avatar=avatar
    ):

        st.markdown(
            msg["content"]
        )

        if (
            msg["role"] == "assistant"
            and "sources" in msg
            and msg["sources"]
        ):

            with st.expander(
                "📚 Sources Juridiques Examen"
            ):

                for src in msg["sources"]:
                    st.markdown(
                        f"- `{src}`"
                    )


if prompt := st.chat_input(
    "Posez votre question juridique "
    "(ex: Quels sont les préavis de licenciement au Togo ?)..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message(
        "user",
        avatar="👤"
    ):
        st.markdown(prompt)

    with st.chat_message(
        "assistant",
        avatar="⚖️"
    ):

        with st.spinner(
            "Analyse des textes de loi & rédaction de la réponse..."
        ):

            try:

                answer, docs = query_pipeline(
                    pipeline,
                    prompt
                )

                sources = list(
                    set(
                        [
                            d.metadata.get(
                                "filename",
                                "Document Officiel"
                            )
                            for d in docs
                        ]
                    )
                )

                st.markdown(answer)

                if sources:

                    with st.expander(
                        "📚 Sources Juridiques Examen"
                    ):

                        for src in sources:
                            st.markdown(
                                f"- `{src}`"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                st.error(
                    f"Une erreur s'est produite lors de l'analyse : {e}"
                )
