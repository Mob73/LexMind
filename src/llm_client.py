from src.config import CONFIG, GEMINI_API_KEY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

def get_llm():
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3.3-70b-instruct",
        temperature=0.2
    )
    return llm

def create_generation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """Vous êtes un assistant juridique spécialisé en droit togolais. 
        Répondez UNIQUEMENT avec le contexte fourni. Citez les articles et sources. Si le contexte n'a aucun rapport avec la quetion de l'utilisateur ou si l'utilisateur parle d'un autre sujet repondez lui normalement et  faites lui comprendre ce que vous etes et en quoi vous etes specialise c'est a dire le droit togolais exclusivement""",),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "Contexte: \n{context}\n\nQuestion: {question}"),
    ])
