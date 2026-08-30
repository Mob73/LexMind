from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import CONFIG


def get_retriever_from_vector_store(vector_store, search_kwargs=None):
    """Retourne un retriever vectoriel simple."""
    if search_kwargs is None:
        search_kwargs = {"k": CONFIG["top_k_initial"]}

    return vector_store.as_retriever(
        search_kwargs=search_kwargs
    )


def create_hybrid_retriever(chunks, vector_store):
    """Crée un retriever hybride BM25 + vectoriel."""

    documents = []

    for chunk in chunks:
        if isinstance(chunk, Document):
            documents.append(chunk)

        elif isinstance(chunk, dict):
            text = (
                chunk.get("page_content")
                or chunk.get("text")
                or chunk.get("content")
                or ""
            )

            metadata = chunk.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            if text:
                documents.append(
                    Document(
                        page_content=str(text),
                        metadata=metadata
                    )
                )

        elif isinstance(chunk, str):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={}
                )
            )

    if not documents:
        raise ValueError(
            "Aucun document valide disponible pour BM25."
        )

    bm25_retriever = BM25Retriever.from_documents(
        documents
    )

    bm25_retriever.k = CONFIG["top_k_initial"]

    vector_retriever = get_retriever_from_vector_store(
        vector_store
    )

    return EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever,
        ],
        weights=CONFIG["hybrid_search_weights"],
    )


def hybrid_retrieve(retriever, query):
    """Exécute simplement la recherche hybride."""
    return retriever.invoke(query)


def evaluate_context_sufficiency(llm, query, documents):
    """Évalue si le contexte est suffisant pour répondre."""

    if not documents:
        return False

    context = "\n\n".join(
        doc.page_content[:500]
        for doc in documents[:3]
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

        return response_text.strip().upper() == "SUFFICIENT"

    except Exception:
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
            doc.page_content[:200]
            for doc in retrieved_docs[:3]
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

    except Exception:
        return original_query


def agentic_retrieve(llm, hybrid_retriever, query):
    """Récupération itérative avec vérification du contexte."""

    iteration = 0
    all_documents = []

    while iteration < CONFIG["max_retrieval_iterations"]:
        iteration += 1

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
            return (
                all_documents[:CONFIG["top_k_final"]],
                True
            )

        if iteration >= CONFIG["max_retrieval_iterations"]:
            return (
                all_documents[:CONFIG["top_k_final"]],
                False
            )

        query = refine_query(
            llm,
            query,
            all_documents
        )

    return (
        all_documents[:CONFIG["top_k_final"]],
        False
    )
