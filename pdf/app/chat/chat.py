import random
from langchain.chat_models import ChatOpenAI
from app.chat.models import ChatArgs
from app.chat.vector_stores import retriever_map
from app.chat.llms import llm_map
from app.chat.memories import memory_map
from app.web.api import set_conversation_components, get_conversation_components
from app.chat.score import random_component_by_score

from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from app.chat.tracing.langfuse import langfuse
from langfuse.model import CreateTrace


def select_component(component_type, component_map, chat_args):
    components = get_conversation_components(chat_args.conversation_id)
    previous_component = components[component_type]
    if previous_component:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args)
    else:
        random_name = random.choice(list(component_map.keys()))
        random_name = random_component_by_score(component_type, component_map)
        builder = component_map[random_name]
        return random_name, builder(chat_args)


def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """
    retriever_name, retriever = select_component("retriever", retriever_map, chat_args)
    lln_name, llm = select_component("llm", llm_map, chat_args)
    condense_question_llm = ChatOpenAI(streaming=False)
    memory_name, memory = select_component("memory", memory_map, chat_args)
    print(
        f"running chain with : memory={memory_name}, retriever={retriever_name}, llm={lln_name}"
    )
    set_conversation_components(
        chat_args.conversation_id,
        retriever=retriever_name,
        llm=lln_name,
        memory=memory_name,
    )

    trace = langfuse.trace(
        CreateTrace(
            id=chat_args.conversation_id,
            metadata=chat_args.metadata,
        )
    )
    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=condense_question_llm,
        memory=memory,
        retriever=retriever,
        callbacks=[trace.getNewHandler()],
    )
