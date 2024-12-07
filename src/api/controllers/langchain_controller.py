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
