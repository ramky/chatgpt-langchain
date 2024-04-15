import os
from pinecone import Pinecone as pinecone

from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings

pc = pinecone(api_key=os.getenv("PINECONE_API_KEY"))

vector_store = Pinecone.from_existing_index(
    os.getenv("PINECONE_INDEX_NAME"), embeddings
)


def build_retriever(chat_args):
    search_kwargs = {
        "filter": {"pdf_id": chat_args.pdf_id},
    }
    return vector_store.as_retriever(search_kwargs=search_kwargs)
