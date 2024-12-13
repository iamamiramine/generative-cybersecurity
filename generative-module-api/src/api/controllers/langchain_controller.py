from fastapi import APIRouter, HTTPException
from typing import Optional, Dict
from enum import Enum

from src.application.langchain.services import langchain_service
from src.domain.models.langchain.langchain_models import (
    PipelineParameters, 
    ContextRetrieverParameters, 
    GenerateResponseParameters, 
    ChainParameters, 
    LoadModelParameters
)
router = APIRouter()

@router.get("/status")
def get_status() -> Dict[str, bool]:
    """Get the current status of the language model system"""
    return {
        "model_loaded": langchain_service.state.is_model_loaded(),
        "pipeline_loaded": langchain_service.state.is_pipeline_loaded(),
        "docs_loaded": langchain_service.state.is_docs_loaded(),
        "retriever_loaded": langchain_service.state.is_retriever_loaded(),
        "chain_loaded": langchain_service.state.is_chain_loaded()
    }


@router.post("/load_model")
def load_model(load_model_parameters: LoadModelParameters):
    """
    Load a HuggingFace model and tokenizer with 4-bit quantization.

    Args:
        model_id (str): Path or name of the model to load. Defaults to WhiteRabbitNeo model.

    Returns:
        dict: Dictionary containing success/failure message

    Example:
        >>> from src.application.langchain.services.langchain_service import load_model
        >>> # Load default model
        >>> load_model()
        >>> # Load specific model
        >>> load_model("models/custom_model")

    Note:
        - Uses 4-bit quantization via BitsAndBytesConfig for memory efficiency
        - Model is loaded in local_files_only mode
        - Sets global _model and _tokenizer variables
    """
    response = langchain_service.load_model(load_model_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response
    
@router.post("/load_pipeline")
def load_pipeline(
    pipeline_parameters: PipelineParameters
) -> dict:
    """
    Load a pipeline for the specified task type.
    
    Args:
        task_type: The type of task for the pipeline (text-generation or text2text-generation)

    Raises:
        Exception: If model is not loaded before calling this function
        ValueError: If an unsupported task type is specified

    Returns:
        dict: Dictionary containing success message

    Example:
        >>> from src.application.langchain.services.langchain_service import load_pipeline
        >>> # First load model using load_model()
        >>> # Then load text generation pipeline
        >>> load_pipeline("text-generation")

    Note:
        - Requires model and tokenizer to be loaded first via load_model()
        - Sets global _hf and _task variables
        - Pipeline is configured with max_new_tokens=4000
    """
    response = langchain_service.load_pipeline(pipeline_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response

@router.post("/load_docs")
def load_docs():
    response = langchain_service.load_docs()
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response


@router.post("/load_ensemble_retriever_from_docs")
def load_ensemble_retriever_from_docs():
    response = langchain_service.load_ensemble_retriever_from_docs()
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response


@router.post("/load_context_retriever")
def load_context_retriever(context_retriever_parameters: ContextRetrieverParameters):
    response = langchain_service.load_context_retriever(context_retriever_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response


@router.post("/load_chain")
def load_chain(chain_parameters: ChainParameters):
    """
    Load a language chain with optional HyDE (Hypothetical Document Embeddings) enhancement.

    Args:
        chain_type (str, optional): The type of chain to load. Defaults to "basic".

    Raises:
        Exception: If model is not loaded before calling this function

    Returns:
        dict: Dictionary containing success message

    Example:
        >>> from src.application.langchain.services.langchain_service import load_chain
        >>> # Load standard RAG chain
        >>> load_chain()
        >>> # Load chain with HyDE
        >>> load_chain(use_hyde=True)

    Note:
        - Requires model to be loaded first via load_model()
        - Sets up either a standard RAG chain or HyDE-enhanced chain
        - Configures system prompts for bash script generation
        - Integrates with Streamlit chat history for memory
    """
    response = langchain_service.load_chain(chain_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response


@router.post("/generate")
def generate(
    generate_response_parameters: GenerateResponseParameters
) -> dict:
    """
    Generate and execute bash commands based on the user's question.
    
    Args:
        question: The user's question/prompt
        max_depth: Maximum number of execution attempts (0=generate only, 1=execute once, >1=allow retries)

    Raises:
        Exception: If model is not loaded before calling this function

    Returns:
        str: The generated response text

    Example:
        >>> from src.application.langchain.services.langchain_service import generate
        >>> # First load model and pipeline
        >>> question = "What are the key principles of cybersecurity?"
        >>> generate(question)

    Note:
        - Requires model and pipeline to be loaded first via load_model() and load_pipeline()
        - Uses a step-by-step reasoning prompt template
        - Saves output to output.txt file
    """
    response = langchain_service.generate(generate_response_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response