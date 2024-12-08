from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.application.langchain.helpers.document_helper import split_documents
from src.application.langchain.services.vector_db_service import create_vector_db

from src.application.langchain.helpers.document_helper import load_txt_files

def ensemble_retriever_from_docs(docs, embeddings=None):
    texts = split_documents(docs)
    vs = create_vector_db(texts, embeddings)
    vs_retriever = vs.as_retriever()

    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever],
        weights=[0.5, 0.5])

    return ensemble_retriever


def load_retriever(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """
    Loads a retriever for enhancing LLM responses through RAG (Retrieval Augmented Generation).
    
    A retriever is a component that searches through a document collection to find relevant context
    for answering questions. It improves LLM performance by:
    1. Providing relevant domain knowledge from trusted sources
    2. Reducing hallucination by grounding responses in actual data
    3. Enabling the LLM to reference specific examples and details
    
    This implementation uses an ensemble retriever that combines:
    - BM25 retrieval: Uses keyword matching for semantic search
    - Vector similarity: Uses embeddings to find conceptually similar content
    
    Args:
        model_name (str): Name of the embedding model to use for vector similarity search
            
    Returns:
        EnsembleRetriever: A retriever that combines BM25 and vector similarity search
    """
    docs = load_txt_files()
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return ensemble_retriever_from_docs(docs, embeddings=embeddings)