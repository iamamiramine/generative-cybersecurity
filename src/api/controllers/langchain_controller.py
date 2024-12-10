from fastapi import APIRouter

from src.application.langchain.services import langchain_service

router = APIRouter()

@router.post("/load_model")
def load_model():
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
    return langchain_service.load_model()
    

@router.post("/load_pipeline")
def load_pipeline(task_type: str):
    """
    Load a HuggingFace pipeline for the specified task type using the loaded model and tokenizer.

    Args:
        task_type (str): Type of task for the pipeline. Currently supports "text-generation" 
            and "text2text-generation".

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
    return langchain_service.load_pipeline(task_type)


@router.post("/load_docs")
def load_docs():
    return langchain_service.load_docs()


@router.post("/load_ensemble_retriever_from_docs")
def load_ensemble_retriever_from_docs():
    return langchain_service.load_ensemble_retriever_from_docs()


@router.post("/load_context_retriever")
def load_context_retriever(use_hyde: bool = False):
    return langchain_service.load_context_retriever(use_hyde)


@router.post("/load_chain")
def load_chain():
    """
    Load a language chain with optional HyDE (Hypothetical Document Embeddings) enhancement.

    Args:
        use_hyde (bool, optional): Whether to use HyDE for improved retrieval. Defaults to False.

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
    return langchain_service.load_chain()


@router.post("/generate")
def generate(question: str):
    """
    Generate a response to a question using the loaded language model pipeline.

    Args:
        question (str): The question to generate a response for

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
    return langchain_service.generate(question)
