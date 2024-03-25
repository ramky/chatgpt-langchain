from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # 1536 embeds - will change in future
from langchain.vectorstores.chroma import Chroma

DEBUG = False

loader = TextLoader("facts.txt")

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
)
docs = loader.load_and_split(text_splitter=text_splitter)

if DEBUG:
    for doc in docs:
        print(doc.page_content)
        print("\n")

db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

prompt = "What is an interesting fact about the English language?"
results = db.similarity_search_with_score(prompt, k=1)

for result in results:
    print("\n")
    # print(result[1])
    print(result[0].page_content)
