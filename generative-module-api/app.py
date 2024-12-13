from fastapi import FastAPI

from src.api.controllers import (
    health_controller,
    langchain_controller,
    local_llm_controller,
)
from src.handlers.exception_handler import add_exception_handlers

tags_metadata = [
    {
        "name": "health",
        "description": "checks the health of the API services",
    },
    {
        "name": "local_llm",
        "description": "Local LLM",
    },
    {
        "name": "langchain",
        "description": "Langchain",
    },
]


app = FastAPI(
    version="1.0",
    title="Generative Cybersecurity API",
    description="API for Generative Cybersecurity",
    openapi_tags=tags_metadata,
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    local_llm_controller.router,
    prefix="/local_llm",
    tags=["local_llm"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    langchain_controller.router,
    prefix="/langchain",
    tags=["langchain"],
    responses={404: {"description": "Not found"}},
)

add_exception_handlers(app=app)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)