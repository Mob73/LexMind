from src.config import CONFIG, GEMINI_API_KEY
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def create_llm():
    if CONFIG["llm_provider"] == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY manquante")
        return GoogleGenerativeAI(
            model=CONFIG["llm_model"],
            temperature=CONFIG["llm_temperature"],
            google_api_key=GEMINI_API_KEY,
            streaming=CONFIG["llm_streaming"],
            callbacks=([StreamingStdOutCallbackHandler()] if CONFIG["llm_streaming"] else [])
        )
    else:
        raise ValueError("Provider non supporté")

def create_generation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """Vous êtes un assistant juridique spécialisé en droit togolais. 
        Répondez UNIQUEMENT avec le contexte fourni. Citez les articles et sources. Si le contexte n'a aucun rapport avec la quetion de l'utilisateur ou si l'utilisateur parle d'un autre sujet repondez lui normalement et  faites lui comprendre ce que vous etes et en quoi vous etes specialise c'est a dire le droit togolais exclusivement""",),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "Contexte: \n{context}\n\nQuestion: {question}"),
    ])