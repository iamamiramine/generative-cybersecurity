from fastapi import APIRouter

from src.application.system.services import system_service

router = APIRouter()

@router.post("/execute_script")
def execute_script(script_number: int = 1):
    return system_service.execute_script(script_number)