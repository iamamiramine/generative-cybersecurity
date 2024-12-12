import torch
import re
import os
import logging
from typing import Dict, List, Any
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.application.langchain.helpers.document_helper import split_documents, load_txt_files
from src.application.langchain.helpers.vector_db_helper import create_vector_db
from src.application.langchain.helpers.retreival_chain_helper import make_rag_chain, make_hyde_chain, prepare_hyde_prompt, prepare_rag_prompt, prepare_basic_prompt
from src.application.langchain.helpers.memory_helper import create_memory_chain
from src.application.langchain.tools.script_execution_tool import ScriptExecutionTool

_model = None
_tokenizer = None
_hf = None
_ensemble_retriever = None
_retrieval_chain = None
_chain = None
_agent = None
_messages = StreamlitChatMessageHistory(key="langchain_messages")
_call_counter = 0

def is_model_loaded() -> bool:
    return _model is not None and _tokenizer is not None

def is_chain_loaded() -> bool:
    return _chain is not None


def load_model(model_id: str = "models/WhiteRabbitNeo_WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"):
    global _model, _tokenizer
    
    if is_model_loaded():
        return {"message": "Model already loaded"}
        
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    
    _tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        local_files_only=True, 
        device_map="auto",
        quantization_config=quantization_config, 
        trust_remote_code=False,
    )
    return {"message": "Model loaded successfully"}


def load_pipeline(task_type: str) -> dict:
    global _hf, _task
    
    if not is_model_loaded():
        raise Exception("Model not loaded. Please call /load-model endpoint first")
        
    # Convert task_type to string since it comes as an Enum
    task_type_str = str(task_type)
    
    if task_type_str not in ["text-generation", "text2text-generation"]:
        raise ValueError(f"Unsupported task type: {task_type_str}. Must be one of: text-generation, text2text-generation")
    
    _task = task_type_str
    pipe = pipeline(
        task_type_str, 
        model=_model, 
        tokenizer=_tokenizer, 
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,  # Lower temperature for more focused responses
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,  # Helps prevent repetitive text
        pad_token_id=_tokenizer.eos_token_id,
        eos_token_id=_tokenizer.eos_token_id,  # Explicitly set end of sequence token
    )
    _hf = HuggingFacePipeline(pipeline=pipe)
    return {"message": f"Pipeline loaded successfully for task: {task_type_str}"}


def load_docs() -> dict:
    global _docs
    _docs = load_txt_files()
    
    return {"message": "Docs loaded successfully"}


def load_ensemble_retriever_from_docs() -> dict:
    global _ensemble_retriever

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    texts = split_documents(_docs)
    vs = create_vector_db(texts, embeddings)
    vs_retriever = vs.as_retriever()

    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

    _ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever],
        weights=[0.5, 0.5])
    
    return {"message": "Ensemble retriever loaded successfully"}


def load_context_retriever(use_hyde: bool = False) -> dict:
    global _retrieval_chain

    if use_hyde:
        hyde_generation_prompt, hyde_final_prompt = prepare_hyde_prompt(_tokenizer)
        _retrieval_chain = make_hyde_chain(_hf, _ensemble_retriever, hyde_generation_prompt, hyde_final_prompt)
    else:
        rag_prompt = prepare_rag_prompt(_tokenizer)
        _retrieval_chain = make_rag_chain(_hf, _ensemble_retriever, rag_prompt)

    return {"message": "Retrieval chain loaded successfully"}


def load_chain(chain_type: str = "basic") -> dict:
    global _chain
    
    if not _hf:
        raise Exception("Pipeline not loaded. Please call /load-pipeline endpoint first")
        
    if chain_type == "basic":
        basic_prompt = prepare_basic_prompt(_tokenizer)
        _chain = create_memory_chain(_hf, None, _messages, basic_prompt)
    elif chain_type == "rag":
        if not _ensemble_retriever:
            raise Exception("Retriever not loaded. Please call /load-retriever endpoint first")
        rag_prompt = prepare_rag_prompt(_tokenizer)
        rag_chain = make_rag_chain(_hf, _ensemble_retriever, rag_prompt)
        _chain = create_memory_chain(_hf, rag_chain, _messages)
    elif chain_type == "hyde":
        if not _ensemble_retriever:
            raise Exception("Retriever not loaded. Please call /load-retriever endpoint first")
        hyde_generation_prompt, hyde_final_prompt = prepare_hyde_prompt(_tokenizer)
        hyde_chain = make_hyde_chain(_hf, _ensemble_retriever, hyde_generation_prompt, hyde_final_prompt)
        _chain = create_memory_chain(_hf, hyde_chain, _messages)
    else:
        raise ValueError(f"Unsupported chain type: {chain_type}")

    return {"message": f"{chain_type} chain loaded successfully"}


def execute_bash_script(script_path: str, timeout: int = 120) -> Dict[str, Any]:
    """Execute a bash script and return its output."""
    executor = ScriptExecutionTool()
    try:
        # Pass the script path directly to the _run method
        result = executor._run(script_path=script_path, timeout=timeout)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "script_path": script_path
        }

def generate(question: str, max_depth: int = 1) -> Dict[str, Any]:
    """Generate and optionally execute bash commands based on the user's question.
    
    Args:
        question (str): The user's question/prompt
        max_depth (int): Controls script execution:
                        0 = generate only, no execution
                        >=1 = generate and execute
    """
    global _call_counter
    _call_counter += 1
    current_call = _call_counter
    
    torch.cuda.empty_cache()
    logging.info(f"\n[USER QUERY {current_call}] {question}")

    if not is_chain_loaded():
        raise Exception("Chain not loaded. Please call /load-chain endpoint first")

    try:
        # Generate script using LLM
        output = _chain.invoke(        
            {"question": question},
            config={"configurable": {"session_id": "foo"}}
        )
        logging.info(f"\n[LLM RESPONSE {current_call}] {output}")

        # Extract code blocks using regex
        code_blocks = re.findall(r'```(?:bash|sh)\n(.*?)\n```', output, re.DOTALL)
        
        # Save the complete response
        with open(f"output_{current_call}.txt", "w") as f:
            f.write(str(output))
        
        # Track execution status for each script
        execution_status: List[Dict[str, Any]] = []
        
        if code_blocks:
            os.makedirs("generated_scripts", exist_ok=True)
            
            for i, code in enumerate(code_blocks):
                script_num = i
                logging.info(f"\n[SCRIPT {current_call}_{script_num:02d}] Processing script...")
                
                # Save the script
                script_content = code if code.strip().startswith('#!/') else f'#!/bin/bash\n\n{code}'
                script_path = f"generated_scripts/command_{current_call}_{script_num:02d}.sh"
                
                with open(script_path, "w") as f:
                    f.write(script_content.strip())
                os.chmod(script_path, 0o755)
                
                status = {
                    "script_number": script_num,
                    "call_number": current_call,
                    "script_content": script_content,
                    "script_path": script_path,
                    "executed_on_kali": False,
                    "execution_result": None
                }
                
                # Execute script if max_depth > 0
                if max_depth > 0:
                    logging.info(f"[SCRIPT {current_call}_{script_num:02d}] Executing on Kali Linux...")
                    execution_result = execute_bash_script(script_path)
                    status.update({
                        "executed_on_kali": True,
                        "execution_result": execution_result,
                        "success": execution_result.get("success", False)
                    })
                    
                    if execution_result.get("success"):
                        logging.info(f"[SCRIPT {current_call}_{script_num:02d}] ✓ Successfully executed")
                    else:
                        error_msg = execution_result.get("error", "Unknown error")
                        logging.error(f"[SCRIPT {current_call}_{script_num:02d}] ✗ Execution failed: {error_msg}")
                else:
                    logging.info(f"[SCRIPT {current_call}_{script_num:02d}] ℹ Script generated but not executed (max_depth=0)")
                
                execution_status.append(status)

        # Save execution results to file
        with open(f"script_results_{current_call}.txt", "w") as f:
            for status in execution_status:
                script_num = status["script_number"]
                f.write(f"Call {current_call}, Script {script_num} results:\n")
                f.write(f"Script content:\n{status['script_content']}\n\n")
                f.write(f"Script path: {status['script_path']}\n")
                f.write(f"Executed on Kali: {status['executed_on_kali']}\n")
                
                if status['executed_on_kali']:
                    result = status['execution_result']
                    f.write(f"Execution success: {result.get('success', False)}\n")
                    if result.get('output'):
                        f.write(f"Output:\n{result['output']}\n")
                    if result.get('error'):
                        f.write(f"Error:\n{result['error']}\n")
                f.write("\n" + "="*50 + "\n\n")

        return {
            "llm_output": output,
            "call_number": current_call,
            "scripts_generated": len(code_blocks),
            "execution_status": execution_status
        }
        
    except Exception as e:
        logging.error(f"Error in generate function: {str(e)}")
        return {
            "error": True,
            "message": str(e),
            "details": "An error occurred during script generation or execution"
        }