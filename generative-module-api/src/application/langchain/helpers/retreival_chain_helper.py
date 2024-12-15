from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import PromptTemplate

from src.application.langchain.models.tokenized_chat_prompt_model import BasicTokenizedChatPromptTemplate, RAGTokenizedChatPromptTemplate, HYDEGenerationPromptTemplate, HYDEFinalPromptTemplate


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def format_docs(docs):
    try:
        print(f"Docs type: {type(docs)}", flush=True)
        # Sanitize each document's content
        sanitized_contents = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                # Remove problematic characters and normalize whitespace
                content = doc.page_content.strip()
                content = content.replace('\r', ' ').replace('\t', ' ')
                sanitized_contents.append(content)
        
        # Join with double newlines
        return "\n\n".join(sanitized_contents)
    except Exception as e:
        print(f"Error formatting docs: {e}", flush=True)
        return ""


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
        print(f"Input type: {type(input)}", flush=True)
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
    return hyde_generation_prompt.get_prompt(), hyde_final_prompt.get_prompt()


def prepare_rag_prompt(tokenizer):
    template = """<|im_start|>You are an expert in bash scripting. Given a question about bash commands, please provide an answer with bash scripts only, and make sure to format with codeblocks using ```bash and ```<|im_end|>.

    <|im_start|>Context: {context}<|im_end|>

    <|im_start|>Question: {question}<|im_end|>

    <|im_start|>Bash Script: """
    custom_rag_prompt = PromptTemplate.from_template(template)
    return custom_rag_prompt


def prepare_basic_prompt(tokenizer):
    basic_prompt = BasicTokenizedChatPromptTemplate(
        tokenizer=tokenizer,
    )
    return basic_prompt.get_prompt()
