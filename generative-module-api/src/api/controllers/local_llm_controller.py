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
    return local_llm_service.download_llm(models_dir, model_filename, model_id)


@router.post("/start_server")
def start_server(
        model_filename: str = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf", 
    ) -> dict:  
    return local_llm_service.start_server(model_filename)


@router.post("/check_server_status")
def check_server_status(
        api_url: str = "http://127.0.0.1:5000",
    ) -> dict:  
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
    return local_llm_service.chat(api_url, messages, temperature, max_new_tokens, top_p, stop)
