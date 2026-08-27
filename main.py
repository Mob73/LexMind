import os
from src.pipeline import initialize_pipeline, index_documents, query_pipeline
from src.config import CONFIG

def main():
    print("\n" + "█"*60)
    print("█      RAG PIPELINE 2026 - TOGOLESE LEGAL AI       █")
    print("█"*60)

    # Vérifier que le dossier documents existe et n'est pas vide
    if not os.path.exists(CONFIG["docs_directory"]):
        os.makedirs(CONFIG["docs_directory"])
        print(f"📁 Dossier '{CONFIG['docs_directory']}' créé. Veuillez y placer vos documents.")
        return

    # Initialiser le pipeline (charge l'index existant)
    pipeline = initialize_pipeline()

    # Si aucun index, proposer d'en créer un
    if not pipeline["is_indexed"]:
        print("Aucun index trouvé. Voulez-vous indexer les documents maintenant ? (o/n)")
        reponse = input().strip().lower()
        if reponse == "o":
            index_documents(pipeline)
        else:
            print("L'application ne peut pas fonctionner sans index. Arrêt.")
            return

    # Mode interactif
    print("\n🤖 Assistant Juridique Togolais prêt. Posez vos questions (tapez 'quit' pour sortir).")
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in ["quit", "exit"]:
            break
        try:
            answer, docs = query_pipeline(pipeline, q)
            print(f"\n📜 Réponse:\n{answer}\n")
            print(f"📚 Sources: {[d.metadata.get('filename') for d in docs]}")
        except Exception as e:
            print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    main()