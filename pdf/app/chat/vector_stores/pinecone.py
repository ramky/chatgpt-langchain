import os
from pinecone import Pinecone as pinecone

from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings

pc = pinecone(api_key=os.getenv("PINECONE_API_KEY"))

vector_store = Pinecone.from_existing_index(
    os.getenv("PINECONE_INDEX_NAME"), embeddings
)
