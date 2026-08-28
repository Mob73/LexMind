from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
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
    """Exécute la recherche hybride."""
    return retriever.invoke(query)

# --- Fonctions pour la récupération agentique ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def evaluate_context_sufficiency(llm, query, documents):
    """Évalue si le contexte est suffisant pour répondre."""
    if not documents:
        return False

    context = "\n\n".join([doc.page_content[:500] for doc in documents[:3]])

    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", """Vous êtes un agent évaluateur. Déterminez si le contexte fourni est SUFFISANT 
        pour répondre à la question de l'utilisateur.
        
        Répondez UNIQUEMENT par:
        - "SUFFICIENT" si le contexte contient toutes les informations nécessaires
        - "INSUFFICIENT" si des informations cruciales manquent
        - "NEED_CLARIFICATION" si la question est ambiguë"""),
        ("human", "Contexte: {context}\n\nQuestion: {question}"),
    ])

    try:
        chain = evaluation_prompt | llm
        response = chain.invoke({"context": context, "question": query, "chat_history": []})
        response_text = response.strip().upper()
        return "SUFFICIENT" in response_text
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        return len(documents) >= CONFIG["top_k_final"]

def refine_query(llm, original_query, retrieved_docs):
    """Reformule la requête pour une meilleure recherche."""
    if not retrieved_docs:
        return original_query

    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system", """Vous êtes un expert en recherche d'information.
        Reformulez la question suivante pour améliorer la recherche de documents.
        Ajoutez des synonymes et des concepts connexes."""),
        ("human", "Question originale: {query}\n\nDocuments déjà trouvés: {context}\n\nNouvelle question:"),
    ])

    try:
        context = "\n".join([doc.page_content[:200] for doc in retrieved_docs[:3]])
        chain = refinement_prompt | llm
        response = chain.invoke({"query": original_query, "context": context, "chat_history": []})
        return response.strip()
    except:
        return original_query

def agentic_retrieve(llm, hybrid_retriever, query):
    """Récupération itérative avec vérification de suffisance du contexte."""
    iteration = 0
    all_documents = []

    while iteration < CONFIG["max_retrieval_iterations"]:
        iteration += 1
        print(f"🔍 Retrieval iteration {iteration}")

        docs = hybrid_retrieve(hybrid_retriever, query)
        all_documents.extend(docs)

        is_sufficient = evaluate_context_sufficiency(
            llm, query, all_documents[:CONFIG["top_k_final"]]
        )

        if is_sufficient:
            print(f"✅ Context sufficient after {iteration} iterations")
            return all_documents[:CONFIG["top_k_final"]], True

        if iteration >= CONFIG["max_retrieval_iterations"]:
            print(f"⚠️ Max iterations reached ({CONFIG['max_retrieval_iterations']})")
            return all_documents[:CONFIG["top_k_final"]], False

        query = refine_query(llm, query, all_documents)
        print(f"🔄 Refined query: {query}")

    return all_documents[:CONFIG["top_k_final"]], False