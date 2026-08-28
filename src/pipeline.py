import pickle
from src.document_loader import load_documents, chunk_documents
from src.vector_store import create_vector_store, load_vector_store, load_chunks
from src.retrievers import create_hybrid_retriever, agentic_retrieve, hybrid_retrieve, get_retriever_from_vector_store
from src.llm_client import create_llm
from src.prompts import create_generation_prompt
from src.config import CONFIG

def initialize_pipeline():
    """Initialise le pipeline en chargeant l'index existant si possible."""
    print("\n" + "="*60)
    print("🚀 INITIALIZING RAG PIPELINE")
    print("="*60)

    llm = create_llm()
    prompt = create_generation_prompt()

    # Essayer de charger l'index existant
    vector_store = load_vector_store()
    hybrid_retriever = None
    is_indexed = False

    if vector_store:
        # Essayer de charger les chunks pour recréer le BM25
        chunks = load_chunks()
        if chunks:
            print("✅ Chunks trouvés, création du retriever hybride...")
            hybrid_retriever = create_hybrid_retriever(chunks, vector_store)
        else:
            print("⚠️ Chunks introuvables, utilisation du retriever vectoriel uniquement.")
            hybrid_retriever = get_retriever_from_vector_store(vector_store)
        is_indexed = True
        print("✅ Index chargé avec succès.")
    else:
        print("⚠️ Aucun index existant trouvé. Veuillez indexer des documents.")

    return {
        "llm": llm,
        "prompt": prompt,
        "hybrid_retriever": hybrid_retriever,
        "vector_store": vector_store,
        "is_indexed": is_indexed,
    }

def index_documents(pipeline_state):
    """Indexe les documents du dossier documents/ et met à jour le pipeline."""
    print("\n" + "="*60)
    print("🔄 STARTING INDEXATION")
    print("="*60)

    docs = load_documents()
    if not docs:
        raise ValueError("Aucun document chargé.")

    chunks = chunk_documents(docs)
    vector_store = create_vector_store(chunks)   # sauvegarde aussi les chunks

    hybrid_retriever = create_hybrid_retriever(chunks, vector_store)

    pipeline_state["vector_store"] = vector_store
    pipeline_state["hybrid_retriever"] = hybrid_retriever
    pipeline_state["is_indexed"] = True

    print("\n✅ Indexation terminée!")
    print(f"   - {len(docs)} documents")
    print(f"   - {len(chunks)} chunks")
    print("="*60)

    return pipeline_state

def query_pipeline(pipeline_state, question):
    """Interroge le pipeline RAG et retourne (réponse, documents)."""
    if not pipeline_state["is_indexed"]:
        raise ValueError("Pipeline non indexé. Appelez index_documents() d'abord.")

    print("\n" + "="*60)
    print(f"❓ Question: {question}")
    print("="*60)

    llm = pipeline_state["llm"]
    hybrid_retriever = pipeline_state["hybrid_retriever"]
    prompt = pipeline_state["prompt"]

    # Récupération des documents
    if CONFIG["enable_agentic_retrieval"]:
        documents, context_sufficient = agentic_retrieve(llm, hybrid_retriever, question)
        print(f"📊 Context sufficient: {context_sufficient}")
    else:
        documents = hybrid_retrieve(hybrid_retriever, question)
        print(f"📊 Retrieved {len(documents)} documents")

    # Génération de la réponse
    print("\n🤖 Generating answer...")
    context = "\n\n---\n\n".join(
        [f"[Source: {doc.metadata.get('filename', 'Inconnu')}]\n{doc.page_content}" for doc in documents]
    )

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": [],
    })
    
    answer = response.content

    print("\n" + "="*60)
    print("✅ Answer generated")
    print("="*60)

    return answer, documents

def simple_query(pipeline_state, question):
    """Version simplifiée : ne retourne que la réponse."""
    answer, _ = query_pipeline(pipeline_state, question)
    return answer
