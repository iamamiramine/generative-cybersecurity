from typing import Union, Annotated

from fastapi import Query
from pydantic import BaseModel

from src.domain.models.base_models import BaseEnum

class TaskType(str, BaseEnum):
    """Available task types for the language model pipeline."""
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"

class LoadModelParameters(BaseModel):
    model_path: Annotated[str, Query(description="The path to the model")] # = "models/WhiteRabbitNeo_WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
    bit_quantization: Annotated[int, Query(description="The bit quantization to use (4 or 8)")] # = 4

class PipelineParameters(BaseModel):
    task_type: Annotated[TaskType, Query(description="The type of task for the pipeline")] = TaskType.TEXT_GENERATION
    max_new_tokens: Annotated[int, Query(description="The maximum number of new tokens to generate")] = 256
    do_sample: Annotated[bool, Query(description="Whether to use sampling")] = True
    temperature: Annotated[float, Query(description="The temperature for sampling")] = 0.7
    top_p: Annotated[float, Query(description="The top-p value for sampling")] = 0.9
    top_k: Annotated[int, Query(description="The top-k value for sampling")] = 50
    repetition_penalty: Annotated[float, Query(description="The repetition penalty for sampling")] = 1.1

class ContextRetrieverParameters(BaseModel):
    use_hyde: Annotated[bool, Query(description="Whether to use HyDE for improved retrieval")] = False

class ChainParameters(BaseModel):
    chain_type: Annotated[str, Query(description="The type of chain to load")] = "basic"

class GenerateResponseParameters(BaseModel):
    question: Annotated[str, Query(description="The user's question/prompt")]
    max_depth: Annotated[int, Query(description="The maximum number of execution attempts (0=generate only, 1=execute once, >1=allow retries)")] = 1
