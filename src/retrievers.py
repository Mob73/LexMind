```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate

from src.config import CONFIG


def get_retriever_from_vector_store(vector_store, search_kwargs=None):
    """Retourne un retriever vectoriel simple."""
    if search_kwargs is None:
        search_kwargs = {"k": CONFIG["top_k_initial"]}

    return vector_store.as_retriever(search_kwargs=search_kwargs)


def create_hybrid_retriever(chunks, vector_store):
    """Crée un retriever hybride (BM25 + Vectoriel)."""
    try:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = CONFIG["top_k_initial"]

        vector_retriever = get_retriever_from_vector_store(vector_store)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=CONFIG["hybrid_search_weights"],
        )

        return ensemble_retriever

    except Exception as e:
        print(f"⚠️ Hybrid retriever not available: {e}")
        print("🔄 Falling back to vector-only retriever")
        return get_retriever_from_vector_store(vector_store)


def hybrid_retrieve(retriever, query):
    """Exécute la recherche avec diagnostic complet dans Streamlit."""
    import streamlit as st

    try:
        st.write("## 🔬 Diagnostic de la recherche")

        st.write(f"**Requête :** `{query}`")
        st.write(
            f"**Retriever :** `{type(retriever).__name__}`"
        )

        # ==================================================
        # RÉCUPÉRER LE VECTOR STORE
        # ==================================================

        vector_store = getattr(
            retriever,
            "vectorstore",
            None
        )

        # Si le retriever est un EnsembleRetriever,
        # chercher le vector store dans ses sous-retrievers.
        if vector_store is None and hasattr(retriever, "retrievers"):
            for sub_retriever in retriever.retrievers:
                vector_store = getattr(
                    sub_retriever,
                    "vectorstore",
                    None
                )

                if vector_store is not None:
                    break

        if vector_store is None:
            st.error(
                "❌ Impossible de récupérer le vector store "
                "depuis le retriever."
            )

            return []

        # ==================================================
        # 1. ÉTAT RÉEL DE LA COLLECTION CHROMA
        # ==================================================

        st.write("### 1️⃣ État réel de la collection Chroma")

        try:
            collection = vector_store._collection

            total_count = collection.count()

            st.write(
                f"**Nombre d'éléments dans la collection : "
                f"{total_count}**"
            )

            sample = collection.get(
                limit=5,
                include=[
                    "documents",
                    "metadatas",
                    "embeddings",
                ],
            )

            sample_ids = sample.get("ids", [])
            sample_documents = sample.get("documents", [])
            sample_metadatas = sample.get("metadatas", [])
            sample_embeddings = sample.get("embeddings", [])

            st.write(
                f"**IDs récupérés pour le test : "
                f"{len(sample_ids)}**"
            )

            st.write(
                f"**Embeddings récupérés pour le test : "
                f"{len(sample_embeddings) if sample_embeddings else 0}**"
            )

            if sample_embeddings:
                first_embedding = sample_embeddings[0]

                st.write(
                    f"**Dimension du premier embedding : "
                    f"{len(first_embedding)}**"
                )

                st.write(
                    f"**Premiers éléments du vecteur : "
                    f"`{first_embedding[:5]}`"
                )

            for i, document in enumerate(sample_documents):
                st.write(f"#### Exemple {i + 1}")

                st.code(
                    document[:500],
                    language=None,
                )

                if i < len(sample_metadatas):
                    st.write(
                        f"**Metadata :** "
                        f"`{sample_metadatas[i]}`"
                    )

        except Exception as e:
            st.error(
                "❌ Erreur inspection collection Chroma : "
                f"`{type(e).__name__}: {e}`"
            )

            st.exception(e)

        # ==================================================
        # 2. RECHERCHE CHROMA BRUTE
        # ==================================================

        st.write("### 2️⃣ Recherche Chroma brute")

        try:
            raw_docs = vector_store.similarity_search(
                query,
                k=5,
            )

            st.write(
                f"**Nombre de résultats Chroma : "
                f"{len(raw_docs)}**"
            )

            for i, doc in enumerate(raw_docs):
                st.write(
                    f"#### Résultat Chroma {i + 1}"
                )

                st.write(
                    f"**Metadata :** `{doc.metadata}`"
                )

                st.code(
                    doc.page_content[:1000],
                    language=None,
                )

        except Exception as e:
            st.error(
                "❌ Erreur pendant similarity_search : "
                f"`{type(e).__name__}: {e}`"
            )

            st.exception(e)

        # ==================================================
        # 3. SCORES DE SIMILARITÉ
        # ==================================================

        st.write("### 3️⃣ Scores de similarité")

        try:
            scored_docs = (
                vector_store
                .similarity_search_with_relevance_scores(
                    query,
                    k=5,
                )
            )

            st.write(
                f"**Résultats avec scores : "
                f"{len(scored_docs)}**"
            )

            for i, (doc, score) in enumerate(scored_docs):
                st.write(
                    f"#### Score {i + 1} : `{score}`"
                )

                st.write(
                    f"**Metadata :** `{doc.metadata}`"
                )

                st.code(
                    doc.page_content[:500],
                    language=None,
                )

        except Exception as e:
            st.error(
                "❌ Impossible de récupérer les scores : "
                f"`{type(e).__name__}: {e}`"
            )

            st.exception(e)

        # ==================================================
        # 4. TEST DU RETRIEVER NORMAL
        # ==================================================

        st.write("### 4️⃣ Résultat du retriever")

        try:
            docs = retriever.invoke(query)

            st.write(
                f"**Nombre de documents retournés : "
                f"{len(docs)}**"
            )

            for i, doc in enumerate(docs):
                st.write(
                    f"#### Document Retriever {i + 1}"
                )

                st.write(
                    f"**Metadata :** `{doc.metadata}`"
                )

                st.code(
                    doc.page_content[:1000],
                    language=None,
                )

            return docs

        except Exception as e:
            st.error(
                "❌ Erreur pendant le retriever : "
                f"`{type(e).__name__}: {e}`"
            )

            st.exception(e)

            return []

    except Exception as e:
        st.error(
            f"❌ Erreur pendant le diagnostic : "
            f"`{type(e).__name__}: {e}`"
        )

        st.exception(e)

        return []


def evaluate_context_sufficiency(llm, query, documents):
    """Évalue si le contexte est suffisant pour répondre."""
    if not documents:
        return False

    context = "\n\n".join(
        [doc.page_content[:500] for doc in documents[:3]]
    )

    evaluation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Vous êtes un évaluateur de contexte juridique.

Déterminez si le contexte fourni permet réellement de répondre à la question.

Répondez UNIQUEMENT par une des trois valeurs suivantes :

SUFFICIENT
INSUFFICIENT
NEED_CLARIFICATION"""
        ),
        (
            "human",
            """Contexte :
{context}

Question :
{question}"""
        ),
    ])

    try:
        chain = evaluation_prompt | llm

        response = chain.invoke({
            "context": context,
            "question": query,
        })

        response_text = response.content

        if isinstance(response_text, list):
            response_text = "".join(
                block.get("text", "")
                for block in response_text
                if isinstance(block, dict)
                and block.get("type") == "text"
            )

        response_text = response_text.strip().upper()

        if response_text == "SUFFICIENT":
            return True

        return False

    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        return False


def refine_query(llm, original_query, retrieved_docs):
    """Reformule la requête pour une meilleure recherche."""
    if not retrieved_docs:
        return original_query

    refinement_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Vous êtes un expert en recherche d'information juridique.

Reformulez la question afin d'améliorer la recherche dans une base de
documents juridiques togolais.

Retournez uniquement la nouvelle requête de recherche."""
        ),
        (
            "human",
            """Question originale :
{query}

Documents déjà trouvés :
{context}

Nouvelle requête de recherche :"""
        ),
    ])

    try:
        context = "\n".join(
            [
                doc.page_content[:200]
                for doc in retrieved_docs[:3]
            ]
        )

        chain = refinement_prompt | llm

        response = chain.invoke({
            "query": original_query,
            "context": context,
        })

        result = response.content

        if isinstance(result, list):
            result = "".join(
                block.get("text", "")
                for block in result
                if isinstance(block, dict)
                and block.get("type") == "text"
            )

        return result.strip()

    except Exception as e:
        print(f"⚠️ Query refinement error: {e}")
        return original_query


def agentic_retrieve(llm, hybrid_retriever, query):
    """Récupération itérative avec vérification du contexte."""
    iteration = 0
    all_documents = []

    while iteration < CONFIG["max_retrieval_iterations"]:
        iteration += 1

        print(f"🔍 Retrieval iteration {iteration}")

        docs = hybrid_retrieve(
            hybrid_retriever,
            query
        )

        all_documents.extend(docs)

        is_sufficient = evaluate_context_sufficiency(
            llm,
            query,
            all_documents[:CONFIG["top_k_final"]]
        )

        if is_sufficient:
            print(
                f"✅ Context sufficient after {iteration} iterations"
            )

            return (
                all_documents[:CONFIG["top_k_final"]],
                True
            )

        if iteration >= CONFIG["max_retrieval_iterations"]:
            print(
                f"⚠️ Max iterations reached "
                f"({CONFIG['max_retrieval_iterations']})"
            )

            return (
                all_documents[:CONFIG["top_k_final"]],
                False
            )

        query = refine_query(
            llm,
            query,
            all_documents
        )

        print(f"🔄 Refined query: {query}")

    return (
        all_documents[:CONFIG["top_k_final"]],
        False
    )
```
