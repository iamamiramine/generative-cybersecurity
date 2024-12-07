from fastapi import APIRouter

from src.application.local_llm.services import llm_server_service

router = APIRouter()


@router.post("/download_llm")
def download_llm(
        models_dir: str = "models", 
        model_filename: str = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf", 
        model_id: str = "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
    ) -> dict:  
    return llm_server_service.download_llm(models_dir, model_filename, model_id)


@router.post("/start_server")
def start_server(
        model_filename: str = "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-f16.gguf", 
    ) -> dict:  
    return llm_server_service.start_server(model_filename)


@router.post("/check_server_status")
def check_server_status(
        api_url: str = "http://127.0.0.1:5000",
    ) -> dict:  
    return llm_server_service.check_server_status(api_url)
