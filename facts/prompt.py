from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import langchain

langchain.debug = True

from redundant_filter_retreiver import RedundantFilterRetriever

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(persist_directory="emb", embedding_function=embeddings)

# Langchain has already done the chaining using RetrievalQA
retriever = RedundantFilterRetriever(embedding=embeddings, chroma=db)

chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

prompt = "What is an interesting fact about the English language?"
# prompt = "What is the longest English word?"
result = chain.run(prompt)

print(result)
