from typing import Any, Dict, List
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from langchain_core.documents import Document


class RedundantFilterRetriever(BaseRetriever):
    embedding: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # calculate embeddings from 'query' string
        emb = self.embedding.embed_query(query)

        # take embeddings and feed them into
        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8  # tolerence for similar documents
        )

    async def aget_relevant_documents(self):
        return []
