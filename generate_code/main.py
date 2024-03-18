from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt)
result = code_chain({"language": "python", "task": "return a list of 10 numbers"})
print(result)
