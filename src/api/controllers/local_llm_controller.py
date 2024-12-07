from fastapi import APIRouter
from typing import Dict, List, Optional, Union

from src.application.local_llm.services import local_llm_service

router = APIRouter()


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
