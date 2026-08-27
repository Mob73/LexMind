from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_generation_prompt():
    """Retourne le prompt de génération pour le RAG."""
    return ChatPromptTemplate.from_messages([
        ("system", """Vous êtes un assistant juridique spécialisé en droit togolais.
        Répondez UNIQUEMENT en utilisant le contexte fourni.
        Si le contexte ne contient pas l'information, dites clairement que vous ne savez pas.
        Citez les articles et les sources quand c'est pertinent.
        Organisez votre réponse de manière claire et structurée."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "Contexte: \n{context}\n\nQuestion: {question}"),
    ])