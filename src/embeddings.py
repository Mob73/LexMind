from langchain_openai import OpenAIEmbeddings

def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("GEMINI_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
