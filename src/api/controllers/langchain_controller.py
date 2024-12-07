from fastapi import APIRouter

from src.domain.exceptions.global_exceptions import GenericException
from src.application.langchain.services import langchain_service

router = APIRouter()


@router.post("/generate")
def generate(question: str):
    try:
        return langchain_service.generate(question)
    except Exception as e:
        raise GenericException(name="LangchainServiceError", message=str(e))
