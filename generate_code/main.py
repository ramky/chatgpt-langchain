from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code: {code}",
    input_variables=["language", "code"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

# result = code_chain({"language": args.language, "task": args.task})

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"],
)

result = chain({"language": args.language, "task": args.task})
print(">>>> GENERATED CODE")
print(result["code"])

print(">>>> GENERATED test")
print(result["test"])
