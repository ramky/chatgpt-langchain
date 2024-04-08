import os
from pinecone import Pinecone as pinecone

from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings

# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment=os.getenv("PINECONE_ENV_NAME"),
# )

# vector_store = Pinecone.from_existing_index(
#     os.getenv("PINECONE_INDEX_NAME"), embeddings
# )

pc = pinecone(api_key=os.getenv("PINECONE_API_KEY"))

print(pc.list_indexes())

vector_store = Pinecone.from_existing_index(
    os.getenv("PINECONE_INDEX_NAME"), embeddings
)
