import os
import requests

from huggingface_hub import hf_hub_download


def download_llm(models_dir, model_filename, model_id) -> dict:
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


def start_server(model_filename) -> dict:
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
        "python", "server.py",
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


def check_server_status(api_url) -> dict:
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
