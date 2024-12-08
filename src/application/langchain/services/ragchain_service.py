from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage


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
        {"question": RunnablePassthrough()},
        RunnablePassthrough.assign(
            # context=RunnableLambda(get_question) | retriever | format_docs,
            context= lambda x: get_question(x) | retriever | format_docs,
            # question=RunnablePassthrough()
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
    )
    
    # Full chain with retrieval
    hyde_chain = (
        hyde_generation
        | RunnablePassthrough.assign(
            context=lambda x: retriever.get_relevant_documents(x) | format_docs,
            hypothetical_document=lambda x: x,
            question=lambda x: get_question(x)
        )
        | final_prompt
        | model
    )

    return hyde_chain
