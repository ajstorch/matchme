from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

PERSISTENCE_DIR = "./chroma_db"


def db_client():
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=PERSISTENCE_DIR,
    )


def singleton_client():
    """
    Returns a singleton client
    """
    global _client
    try:
        return _client
    except NameError:
        _client = db_client()
        return _client
