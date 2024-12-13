import requests
from typing import Dict, List, Optional, Union
import os

from huggingface_hub import hf_hub_download


def download_llm(
        models_dir: str = "models", 
        model_filename: str = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf", 
        model_id: str = "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
    ) -> dict:
    """
    Download a model from Hugging Face Hub if not already present locally. Models are available at https://huggingface.co/api/models

    Args:
        models_dir (str): Directory path where models should be stored
        model_filename (str): Name of the model file to download (e.g. 'model.gguf')
        model_id (str): Hugging Face model ID (e.g. 'organization/model-name')

    Example:
        >>> from application.local_llm.services.llm_server_service import download_llm
        >>> models_dir = "models"
        >>> model_filename = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf" 
        >>> model_id = "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
        >>> download_llm(models_dir, model_filename, model_id)

    Returns:
        dict: Dictionary containing the message and the path to the downloaded model file

    Raises:
        Exception: If model download fails
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, model_filename)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        try:
            print(f"Downloading {model_filename} from {model_id}")
            
            # Download the model using huggingface_hub
            hf_hub_download(
                repo_id=model_id,
                filename=model_filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"Model downloaded successfully to {models_dir}")
            
        except Exception as e:
            raise Exception(f"Failed to download model: {str(e)}")

    return {"Message": f"Model downloaded successfully at {model_path}"}


def start_server(
        model_filename: str = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf"
    ) -> dict:
    """
    Start the text-generation-webui server with the specified model.

    Args:
        model_filename (str): Path to the model file to load

    Raises:
        Exception: If server fails to start

    Example:
        >>> from application.local_llm.services.llm_server_service import start_server
        >>> model_filename = "models/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf"
        >>> start_server(model_filename)

    Returns:
        dict: Dictionary containing the server status

    Note:
        - Requires text-generation-webui to be installed in the current environment
        - Server is started asynchronously with a 10 second wait for startup
    """
    import subprocess
    
    server_command = [
        "python", "text-generation-webui/server.py",
        "--model", model_filename,
        "--api"
    ]
    
    try:
        subprocess.Popen(server_command)
        print(f"Server starting with model: {model_filename}")
        # Wait for server to start
        import time
        time.sleep(10)  # Adjust as needed
    except Exception as e:
        raise Exception(f"Failed to start server: {str(e)}")

    return {"Message": f"Server started with model: {model_filename}"}


def check_server_status(
        api_url: str = "http://127.0.0.1:5000"
    ) -> dict:
    """
    Check if the text-generation-webui server is running and responsive.

    Args:
        api_url (str): Base URL of the server API endpoint

    Example:
        >>> from application.local_llm.services.llm_server_service import check_server_status
        >>> api_url = "http://localhost:8000"
        >>> check_server_status(api_url)
    
    Returns:
        dict: Dictionary containing the server status
    """
    try:
        response = requests.get(f"{api_url}/v1/models")
        return {"Status": "Server is running"} if response.status_code == 200 else {"Status": "Server is not running"}
    except requests.exceptions.RequestException:
        return {"Status": "Server is not running"}


def generate(
        api_url: str = "http://127.0.0.1:5000",
        prompt: str = "",
        temperature: Optional[float] = 0.7,
        max_new_tokens: Optional[int] = 512,
        top_p: Optional[float] = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Union[str, float]]:
    """
    Generate text based on the prompt.
    
    Args:
        prompt (str): Input prompt for the model
        temperature (float, optional): Override default temperature
        max_new_tokens (int, optional): Override default max_new_tokens
        top_p (float, optional): Override default top_p
        stop (List[str], optional): List of stop sequences

    Example:
        >>> from application.local_llm.services.local_llm_service import generate
        >>> generate(api_url="http://127.0.0.1:5000", prompt="Hello, how are you?", temperature=0.7, max_new_tokens=512, top_p=0.9, stop=["\n"])
        
    Returns:
        Dict containing generated text and metadata
    """
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "stop": stop or [],
        "stream": False
    }

    try:
        response = requests.post(
            f"{api_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"API returned status code {response.status_code}")
            
        result = response.json()
        return {
            "text": result["choices"][0]["text"],
            "finish_reason": result["choices"][0]["finish_reason"],
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0)
        }
        
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"API request failed: {str(e)}")
        

def chat(
        api_url: str = "http://127.0.0.1:5000",
        messages: List[Dict[str, str]] = [],
        temperature: Optional[float] = 0.7,
        max_new_tokens: Optional[int] = 512,
        top_p: Optional[float] = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Union[str, float]]:
    """
    Chat with the model using a list of messages.
    
    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
        temperature (float, optional): Override default temperature
        max_new_tokens (int, optional): Override default max_new_tokens
        top_p (float, optional): Override default top_p
        stop (List[str], optional): List of stop sequences

    Example:
        >>> from application.local_llm.services.local_llm_service import chat
        >>> chat(api_url="http://127.0.0.1:5000", messages=[{"role": "user", "content": "Hello, how are you?"}], temperature=0.7, max_new_tokens=512, top_p=0.9, stop=["\n"])
        
    Returns:
        Dict containing generated response and metadata
    """
    try:
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "stop": stop or [],
                "stream": False
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"API returned status code {response.status_code}")
            
        result = response.json()
        return {
            "message": result["choices"][0]["message"],
            "finish_reason": result["choices"][0]["finish_reason"],
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0)
        }
        
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"API request failed: {str(e)}")