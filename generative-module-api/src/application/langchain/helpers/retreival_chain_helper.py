from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from src.application.langchain.models.tokenized_chat_prompt_model import BasicTokenizedChatPromptTemplate, RAGTokenizedChatPromptTemplate, HYDEGenerationPromptTemplate, HYDEFinalPromptTemplate
from src.application.database.helpers.document_helper import format_docs

def find_similar(vs, query):
    """
    Find documents similar to the given query using vector store similarity search.
    
    Args:
        vs: Vector store instance to search in
        query (str): Query text to find similar documents for
        
    Returns:
        list: List of documents that are semantically similar to the query,
              ordered by similarity score
    """
    docs = vs.similarity_search(query)
    return docs


def get_question(input):
    """
    Extract the question text from various input types.
    
    Args:
        input: The input to extract the question from. Can be:
            - None: Returns None
            - str: Returns the string directly
            - dict: Returns the value of the 'question' key
            - BaseMessage: Returns the message content
            
    Returns:
        str or None: The extracted question text, or None if input is None
        
    Raises:
        Exception: If input is not one of the supported types
    """
    # Return None for empty input
    if not input:
        return None
    # Return string input directly    
    elif isinstance(input,str):
        return input
    # Extract question from dict if it has 'question' key
    elif isinstance(input,dict) and 'question' in input:
        return input['question']
    # Get content from BaseMessage objects
    elif isinstance(input,BaseMessage):
        return input.content
    # Raise exception for unsupported input types
    else:
        print(f"Input type: {type(input)}", flush=True)
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(model, retriever, rag_prompt):
    """
    Create a Retrieval-Augmented Generation (RAG) chain that combines document retrieval with LLM generation.
    
    Args:
        model: The language model to use for generation
        retriever: Document retriever component to fetch relevant context
        rag_prompt: Prompt template for RAG generation
        
    Returns:
        RunnableSequence: A chain that:
            1. Takes a question as input
            2. Retrieves relevant documents as context
            3. Formats the context and question into a prompt
            4. Generates a response using the model
    """
    # Create RAG chain by composing components:
    rag_chain = (
        # Take raw question input
        {"question": RunnablePassthrough()}
        # Retrieve and format context, extract clean question
        | RunnablePassthrough.assign(
            context= lambda x: format_docs(retriever.get_relevant_documents(get_question(x))),
            question=lambda x: get_question(x)
        )
        # Format prompt with context and question
        | rag_prompt
        # Generate response using model
        | model
    )

    return rag_chain


def make_hyde_chain(model, retriever, hyde_generation_prompt, final_prompt):
    """
    Create a Hypothetical Document Embeddings (HYDE) chain that uses a two-stage retrieval process.
    
    Args:
        model: The language model to use for generation
        retriever: Document retriever component to fetch relevant context
        hyde_generation_prompt: Prompt template for generating hypothetical documents
        final_prompt: Prompt template for final response generation
        
    Returns:
        RunnableSequence: A chain that:
            1. Generates a hypothetical document based on the question
            2. Performs first retrieval using the hypothetical document
            3. Performs second retrieval using the original question
            4. Combines contexts and generates final response
    """
    # First stage: Generate hypothetical document from question
    hyde_generation = (
        {"question": RunnablePassthrough()} 
        | hyde_generation_prompt
        | model
    )

    # Second stage: First retrieval using hypothetical document
    first_retrieval = (
        hyde_generation
        | RunnablePassthrough.assign(
            # Get context using hypothetical document
            context=lambda x: format_docs(retriever.get_relevant_documents(x)),
            # Store hypothetical document for later use
            hypothetical_document=lambda x: x,
            # Extract clean question
            question=lambda x: get_question(x)
        )
    )
    
    # Third stage: Second retrieval and final response generation
    second_retrieval = (
        first_retrieval
        | RunnablePassthrough.assign(
            # Get additional context using original question
            additional_context=lambda x: format_docs(
                retriever.get_relevant_documents(get_question(x["question"]))
            )
        )
        | {
            # Combine both contexts
            "context": lambda x: f"{x['context']}\n\n{x['additional_context']}",
            # Pass through original question
            "question": lambda x: x["question"],
            # Pass through hypothetical document
            "hypothetical_document": lambda x: x["hypothetical_document"]
        }
        # Format final prompt and generate response
        | final_prompt
        | model
    )

    return second_retrieval


def prepare_hyde_prompt(tokenizer):
    """
    Prepare HYDE (Hypothetical Document Embeddings) prompts for generation and final response.
    
    Args:
        tokenizer: The tokenizer to use for processing prompts
        
    Returns:
        tuple: A tuple containing:
            - The HYDE generation prompt template for creating hypothetical documents
            - The HYDE final prompt template for generating the final response
    """
    # Create prompt template for generating hypothetical documents
    hyde_generation_prompt = HYDEGenerationPromptTemplate(
        tokenizer=tokenizer,
    )
    
    # Create prompt template for final response generation
    hyde_final_prompt = HYDEFinalPromptTemplate(
        tokenizer=tokenizer,
    )
    
    # Return both prompt templates
    return hyde_generation_prompt.get_prompt(), hyde_final_prompt.get_prompt()


def prepare_rag_prompt(tokenizer):
    """
    Prepare a RAG (Retrieval Augmented Generation) prompt template for generating responses.
    
    Args:
        tokenizer: The tokenizer to use for processing prompts
        
    Returns:
        PromptTemplate: A formatted template that combines:
            - System message defining the bash expert role
            - Retrieved context placeholder
            - Question placeholder 
            - Answer prefix
    """
    # Create RAG prompt template with provided tokenizer
    rag_prompt = RAGTokenizedChatPromptTemplate(
        tokenizer=tokenizer,
    )
    
    # Return the formatted prompt template
    return rag_prompt.get_prompt()


def prepare_basic_prompt(tokenizer):
    """
    Prepare a basic prompt template for simple question-answer interactions.
    
    Args:
        tokenizer: The tokenizer to use for processing prompts
        
    Returns:
        PromptTemplate: A formatted template that combines:
            - System base message from config
            - Question placeholder
            - Answer prefix for bash scripts
    """
    # Create basic prompt template with provided tokenizer
    basic_prompt = BasicTokenizedChatPromptTemplate(
        tokenizer=tokenizer,
    )
    
    # Return the formatted prompt template
    return basic_prompt.get_prompt()
