from fastapi import APIRouter
from typing import Dict, List, Optional, Union

from src.application.local_llm.services import local_llm_service

router = APIRouter()


@router.post("/download_llm")
def download_llm(
        models_dir: str = "models", 
        model_filename: str = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf", 
        model_id: str = "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
    ) -> dict:
    """Downloads a language model from Hugging Face Hub.
    
    This function downloads a specified language model from Hugging Face Hub and saves it
    to the local filesystem. The model can then be used for local inference.
    
    Args:
        models_dir (str, optional): Directory where the model will be saved. 
            Defaults to "models".
        model_filename (str, optional): Filename to save the model as.
            Defaults to "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf".
        model_id (str, optional): Hugging Face model ID to download.
            Defaults to "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B".
            
    Returns:
        dict: A dictionary containing status information about the download.
            Example: {"status": "success", "message": "Model downloaded successfully"}
            
    Example:
        >>> download_llm()  # Downloads default model
        >>> download_llm("custom_models", "my_model.gguf", "organization/model-name")
    """
    return local_llm_service.download_llm(models_dir, model_filename, model_id)


@router.post("/start_server")
def start_server(
        model_filename: str = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf", 
    ) -> dict:  
    """Starts a local LLM server using the specified model.
    
    This function initializes and starts a local language model server using the specified
    model file. The server will be available for making inference requests once started.
    
    Args:
        model_filename (str, optional): Filename of the model to load.
            Defaults to "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf".
            
    Returns:
        dict: A dictionary containing status information about the server start.
            Example: {"status": "success", "message": "Server started successfully"}
            
    Example:
        >>> start_server()  # Starts server with default model
        >>> start_server("custom_model.gguf")  # Starts server with custom model
    """
    return local_llm_service.start_server(model_filename)


@router.post("/check_server_status")
def check_server_status(
        api_url: str = "http://127.0.0.1:5000",
    ) -> dict:  
    """Checks the status of a local LLM server.
    
    This function verifies if a local language model server is running and accessible
    at the specified API URL. It can be used to ensure the server is ready to handle
    inference requests before attempting to use it.
    
    Args:
        api_url (str, optional): URL where the LLM server is running.
            Defaults to "http://127.0.0.1:5000".
            
    Returns:
        dict: A dictionary containing status information about the server.
            Example: {"status": "success", "message": "Server is running"}
            or {"status": "error", "message": "Server not responding"}
            
    Example:
        >>> check_server_status()  # Checks default localhost server
        >>> check_server_status("http://192.168.1.100:5000")  # Checks remote server
    """
    return local_llm_service.check_server_status(api_url)


@router.post("/generate")
def generate(
        api_url: str = "http://127.0.0.1:5000",
        prompt: str = "",
        temperature: Optional[float] = 0.7,
        max_new_tokens: Optional[int] = 512,
        top_p: Optional[float] = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Union[str, float]]:
    """Generates text using a local LLM server.
    
    This function sends a prompt to a local language model server and returns the generated text response.
    It allows customization of generation parameters like temperature and token limits.
    
    Args:
        api_url (str, optional): URL where the LLM server is running.
            Defaults to "http://127.0.0.1:5000".
        prompt (str, optional): The input text prompt for generation.
            Defaults to empty string.
        temperature (float, optional): Controls randomness in generation. Higher values (e.g. 1.0) make output more random,
            lower values (e.g. 0.1) make it more deterministic. Defaults to 0.7.
        max_new_tokens (int, optional): Maximum number of tokens to generate.
            Defaults to 512.
        top_p (float, optional): Nucleus sampling parameter. Only tokens with cumulative probability < top_p are kept
            for generation. Defaults to 0.9.
        stop (List[str], optional): List of strings that will stop generation when encountered.
            Defaults to None.
            
    Returns:
        Dict[str, Union[str, float]]: A dictionary containing the generated text and metadata.
            Example: {
                "text": "Generated response text",
                "tokens_generated": 125,
                "generation_time": 2.3
            }
            
    Example:
        >>> # Basic generation with default parameters
        >>> generate(prompt="Write a poem about AI")
        
        >>> # Generation with custom parameters
        >>> generate(
        ...     prompt="Explain quantum computing",
        ...     temperature=0.5,
        ...     max_new_tokens=1024,
        ...     top_p=0.95,
        ...     stop=["END"]
        ... )
    """
    return local_llm_service.generate(api_url, prompt, temperature, max_new_tokens, top_p, stop)


@router.post("/chat")
def chat(
        api_url: str = "http://127.0.0.1:5000",
        messages: List[Dict[str, str]] = [],
        temperature: Optional[float] = 0.7,
        max_new_tokens: Optional[int] = 512,
        top_p: Optional[float] = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Union[str, float]]:
    """
    Send a chat message to a local language model server and get a response.
    This function handles multi-turn conversations by accepting a list of previous messages
    along with the current message.

    Args:
        api_url (str, optional): URL where the LLM server is running.
            Defaults to "http://127.0.0.1:5000".
        messages (List[Dict[str, str]], optional): List of previous conversation messages.
            Each message should be a dict with "role" and "content" keys.
            Example: [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
            Defaults to empty list.
        temperature (float, optional): Controls randomness in generation. Higher values (e.g. 1.0) make output more random,
            lower values (e.g. 0.1) make it more deterministic. Defaults to 0.7.
        max_new_tokens (int, optional): Maximum number of tokens to generate in response.
            Defaults to 512.
        top_p (float, optional): Nucleus sampling parameter. Only tokens with cumulative probability < top_p are kept
            for generation. Defaults to 0.9.
        stop (List[str], optional): List of strings that will stop generation when encountered.
            Defaults to None.

    Returns:
        Dict[str, Union[str, float]]: A dictionary containing the generated response and metadata.
            Example: {
                "text": "I'm doing well, thank you for asking!",
                "tokens_generated": 8,
                "generation_time": 0.5
            }

    Example:
        >>> # Single message chat
        >>> chat(messages=[{"role": "user", "content": "Hello!"}])

        >>> # Multi-turn conversation
        >>> chat(
        ...     messages=[
        ...         {"role": "user", "content": "What is Python?"},
        ...         {"role": "assistant", "content": "Python is a programming language."},
        ...         {"role": "user", "content": "What makes it popular?"}
        ...     ],
        ...     temperature=0.8,
        ...     max_new_tokens=1024
        ... )
    """
    return local_llm_service.chat(api_url, messages, temperature, max_new_tokens, top_p, stop)
