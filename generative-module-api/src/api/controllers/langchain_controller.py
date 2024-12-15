from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from src.application.langchain.services import langchain_service
from src.domain.models.langchain.langchain_models import (
    PipelineParameters, 
    EnsembleRetrieverParameters, 
    GenerateResponseParameters, 
    ChainParameters, 
    LoadModelParameters
)
router = APIRouter()


@router.post("/load_model")
def load_model(load_model_parameters: LoadModelParameters) -> dict:
    """
    Load a language model using the provided parameters.
    
    This endpoint loads a language model from HuggingFace Hub with the specified parameters.
    If there is an error during model loading, it raises an HTTPException with the error details.

    Args:
        load_model_parameters (LoadModelParameters): Parameters for loading the model, including:
            - model_name (str): Name/path of the HuggingFace model to load
            - model_kwargs (dict, optional): Additional keyword arguments for model loading
            - tokenizer_kwargs (dict, optional): Additional keyword arguments for tokenizer loading

    Returns:
        dict: A dictionary containing the response message indicating successful model loading
            Example: {"message": "Model loaded successfully"}

    Raises:
        HTTPException: If there is an error during model loading, with status code 400
            Example: HTTPException(status_code=400, detail="Failed to load model: <error details>")

    Example:
        >>> parameters = LoadModelParameters(
        ...     model_name="WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
        ... )
        >>> load_model(parameters)
        {"message": "Model loaded successfully"}
    """
    response = langchain_service.load_model(load_model_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response

@router.post("/load_pipeline")
def load_pipeline(pipeline_parameters: PipelineParameters) -> dict:
    """
    Load a text generation pipeline with the specified parameters.
    
    This endpoint configures and loads a HuggingFace pipeline for text generation using the
    previously loaded model and tokenizer. The pipeline is configured with parameters like
    temperature, top_p, top_k etc. that control the generation behavior.

    Args:
        pipeline_parameters (PipelineParameters): Parameters for configuring the pipeline, including:
            - max_new_tokens (int): Maximum number of tokens to generate
            - do_sample (bool): Whether to use sampling for generation
            - temperature (float): Controls randomness in generation (higher = more random)
            - top_p (float): Nucleus sampling parameter
            - top_k (int): Top-k sampling parameter
            - repetition_penalty (float): Penalty for repeating tokens
            - task_type (str, optional): Type of task for the pipeline (default is "text-generation")

    Returns:
        dict: A dictionary containing the response message indicating successful pipeline loading
            Example: {"message": "Pipeline loaded successfully for task: text-generation"}

    Raises:
        HTTPException: If there is an error during pipeline loading, with status code 400
            Example: HTTPException(status_code=400, detail="Failed to load pipeline: <error details>")

    Example:
        >>> parameters = PipelineParameters(
        ...     max_new_tokens=256,
        ...     do_sample=True,
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     top_k=50,
        ...     repetition_penalty=1.1
        ... )
        >>> load_pipeline(parameters)
        {"message": "Pipeline loaded successfully for task: text-generation"}
    """
    response = langchain_service.load_pipeline(pipeline_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response

@router.post("/load_docs")
def load_docs() -> dict:
    """
    Load documents into the langchain service.
    
    This endpoint loads documents from a predefined source into the langchain service
    for further processing and retrieval. The documents are processed and stored in a format
    that can be used by the ensemble retriever and other langchain components.

    Returns:
        dict: A dictionary containing the response message indicating successful document loading
            Example: {"message": "Documents loaded successfully", "num_docs": 42}

    Raises:
        HTTPException: If there is an error during document loading, with status code 400
            Example: HTTPException(status_code=400, detail="Failed to load documents: <error details>")

    Example:
        >>> load_docs()
        {"message": "Documents loaded successfully", "num_docs": 42}
    """
    response = langchain_service.load_docs()
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response

@router.post("/load_ensemble_retriever_from_docs")
def load_ensemble_retriever_from_docs(ensemble_retriever_parameters: EnsembleRetrieverParameters) -> dict:
    """
    Load an ensemble retriever using the provided parameters and previously loaded documents.
    
    This endpoint creates and configures an ensemble retriever that combines multiple retrieval methods
    to effectively search through the loaded documents. The ensemble retriever can use different 
    strategies like BM25, embedding similarity, or hybrid approaches based on the provided parameters.

    Args:
        ensemble_retriever_parameters (EnsembleRetrieverParameters): Parameters for configuring the 
            ensemble retriever, including weights for different retrievers, search parameters, and 
            other configuration options.

    Returns:
        dict: A dictionary containing the response message indicating successful retriever loading
            Example: {"message": "Ensemble retriever loaded successfully"}

    Raises:
        HTTPException: If there is an error during retriever loading, with status code 400
            Example: HTTPException(status_code=400, detail="Failed to load ensemble retriever: <error details>")

    Example:
        >>> parameters = EnsembleRetrieverParameters(
        ...     bm25_weight=0.5,
        ...     embedding_weight=0.5,
        ...     k=5
        ... )
        >>> load_ensemble_retriever_from_docs(parameters)
        {"message": "Ensemble retriever loaded successfully"}
    """
    response = langchain_service.load_ensemble_retriever_from_docs(ensemble_retriever_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response

@router.post("/load_chain")
def load_chain(chain_parameters: ChainParameters) -> dict:
    """
    Load and configure a language model chain using the provided parameters.
    
    This endpoint creates and configures a language model chain that will be used for generating
    responses. The chain can be configured with different prompts, models, and memory settings
    based on the provided parameters.

    Args:
        chain_parameters (ChainParameters): Parameters for configuring the language model chain,
            including prompt templates, model settings, memory configuration, and other options.

    Returns:
        dict: A dictionary containing the response message indicating successful chain loading
            Example: {"message": "Chain loaded successfully"}

    Raises:
        HTTPException: If there is an error during chain loading, with status code 400
            Example: HTTPException(status_code=400, detail="Failed to load chain: <error details>")

    Example:
        >>> parameters = ChainParameters(
        ...     prompt_template="Answer the question: {question}",
        ...     model_name="gpt-3.5-turbo",
        ...     use_memory=True
        ... )
        >>> load_chain(parameters)
        {"message": "Chain loaded successfully"}
    """
    response = langchain_service.load_chain(chain_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response

@router.post("/generate")
def generate(generate_response_parameters: GenerateResponseParameters) -> Dict[str, Any]:
    """
    Generate a response using the configured language model chain.
    
    This endpoint uses the previously configured language model chain to generate responses based on 
    the provided input parameters. It leverages the loaded model, retriever, and chain settings to 
    produce contextually relevant responses.

    Args:
        generate_response_parameters (GenerateResponseParameters): Parameters for generating the response,
            including the input text/query, generation settings, and any additional configuration options.

    Returns:
        Dict[str, Any]: A dictionary containing the generated response and any additional metadata
            Example: {
                "response": "Generated text response",
                "sources": ["Source document 1", "Source document 2"],
                "metadata": {...}
            }

    Raises:
        HTTPException: If there is an error during generation, with status code 400
            Example: HTTPException(status_code=400, detail="Failed to generate response: <error details>")

    Example:
        >>> parameters = GenerateResponseParameters(
        ...     query="What are common cybersecurity threats?",
        ...     max_tokens=256,
        ...     temperature=0.7
        ... )
        >>> generate(parameters)
        {
            "response": "Common cybersecurity threats include...",
            "sources": ["Security report 2023", "Threat analysis doc"],
            "metadata": {"confidence": 0.95, "processing_time": "1.2s"}
        }
    """
    response = langchain_service.generate(generate_response_parameters)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response