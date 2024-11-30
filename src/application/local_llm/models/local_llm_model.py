import requests
import json
from typing import Dict, List, Optional, Union

class LocalLLMAgent:
    def __init__(
        self, 
        api_url: str = "http://127.0.0.1:5000",
        context_size: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ):
        """
        Initialize the Local LLM Agent.
        
        Args:
            api_url (str): URL of the text-generation-webui API
            context_size (int): Maximum context size for the model
            temperature (float): Sampling temperature (0.0 to 2.0)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            max_new_tokens (int): Maximum number of tokens to generate
        """
        self.api_url = api_url.rstrip('/')
        self.context_size = context_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens


    def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
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
            
        Returns:
            Dict containing generated text and metadata
        """
        payload = {
            "prompt": prompt,
            "temperature": temperature or self.temperature,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "top_p": top_p or self.top_p,
            "stop": stop or [],
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.api_url}/v1/completions",
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
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
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
            
        Returns:
            Dict containing generated response and metadata
        """
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": temperature or self.temperature,
                    "max_new_tokens": max_new_tokens or self.max_new_tokens,
                    "top_p": top_p or self.top_p,
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