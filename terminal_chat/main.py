# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    FileChatMessageHistory,
)

import os
import sys

chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
# memory = ConversationBufferMemory(
# chat_memory=FileChatMessageHistory("messages.json"),
memory = ConversationSummaryMemory(
    memory_key="messages", return_messages=True, llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    print("What would you like to do?  Type exit to terminate.")
    content = input(">> ")
    if content == "exit":
        sys.exit(0)

    result = chain({"content": content})
    print(result["text"])
