import requests
from typing import Dict, List, Optional, Union


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