from langchain.llms import OpenAI
import os

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
result = llm("Write a very short poem")
print(result)
