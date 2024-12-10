from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from src.application.langchain.models.tokenized_chat_prompt_model import BasicTokenizedChatPromptTemplate, RAGTokenizedChatPromptTemplate, HYDEGenerationPromptTemplate, HYDEFinalPromptTemplate


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(input):
    if not input:
        return None
    elif isinstance(input,str):
        return input
    elif isinstance(input,dict) and 'question' in input:
        return input['question']
    elif isinstance(input,BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(model, retriever, rag_prompt):
    rag_chain = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            context= lambda x: format_docs(retriever.get_relevant_documents(get_question(x))),
            question=lambda x: get_question(x)
        )
        | rag_prompt
        | model
    )

    return rag_chain


def make_hyde_chain(model, retriever, hyde_generation_prompt, final_prompt):
    # Generate hypothetical document
    hyde_generation = (
        {"question": RunnablePassthrough()} 
        | hyde_generation_prompt
        | model
        | RunnableLambda(lambda x: {"question": get_question(x), "hypothetical_document": x})  # Convert to dict
    )
    
    # Full chain with retrieval
    hyde_chain = (
        hyde_generation
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.get_relevant_documents(get_question(x["question"]))),
        )
        | final_prompt
        | model
    )

    return hyde_chain


def prepare_hyde_prompt(tokenizer):
    hyde_generation_prompt = HYDEGenerationPromptTemplate(
        tokenizer=tokenizer,
    )
    hyde_final_prompt = HYDEFinalPromptTemplate(
        tokenizer=tokenizer,
    )
    return hyde_generation_prompt, hyde_final_prompt


def prepare_rag_prompt(tokenizer):
    rag_prompt = RAGTokenizedChatPromptTemplate(
        tokenizer=tokenizer,
    )
    return rag_prompt


def prepare_basic_prompt(tokenizer):
    basic_prompt = BasicTokenizedChatPromptTemplate(
        tokenizer=tokenizer,
    )
    return basic_prompt
