from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # 1536 embeds - will change in future

DEBUG = True

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
