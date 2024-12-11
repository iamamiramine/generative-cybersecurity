import torch
import re
import os
import logging
from typing import Dict, List, Any
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent import AgentExecutor

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from src.application.system.services.system_service import execute_script

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
        
    if task_type not in ["text-generation", "text2text-generation", "summarization", "translation"]:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    _task = task_type
    pipe = pipeline(
        task_type, 
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
    return {"message": "Pipeline loaded successfully"}


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

    # Initialize the agent with our script execution tool
    tools = [ScriptExecutionTool()]
    
    # Create an agent that can decide when to use the tool
    agent = initialize_agent(
        tools,
        _hf,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    # Store the agent in the global scope
    global _agent
    _agent = agent
        
    return {"message": f"{chain_type} chain loaded successfully"}


def generate(question: str, max_depth: int = 1) -> Dict[str, Any]:
    """Generate and optionally execute bash commands based on the user's question.
    
    Args:
        question (str): The user's question/prompt
        max_depth (int): Maximum number of times the LLM can call itself or execute scripts.
                        0 = generate only, no execution
                        1 = execute once
                        >1 = allow multiple executions for debugging
    """
    global _call_counter
    _call_counter += 1
    current_call = _call_counter
    
    torch.cuda.empty_cache()
    logging.info(f"\n[USER QUERY {current_call}] {question}")

    if not is_chain_loaded():
        raise Exception("Chain not loaded. Please call /load-chain endpoint first")

    try:
        # First, get the bash commands from the LLM
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
        
        # If max_depth is 0, only generate scripts without executing
        if max_depth == 0:
            if code_blocks:
                os.makedirs("generated_scripts", exist_ok=True)
                for i, code in enumerate(code_blocks):
                    script_num = i
                    filename = f"generated_scripts/command_{current_call}_{script_num:02d}.sh"
                    with open(filename, "w") as f:
                        f.write(code.strip())
                    execution_status.append({
                        "script_number": script_num,
                        "call_number": current_call,
                        "executed_on_kali": False,
                        "success": True,
                        "message": "Script generated but not executed (max_depth=0)",
                        "script_path": filename
                    })
            return {
                "llm_output": output,
                "call_number": current_call,
                "scripts_generated": len(code_blocks),
                "execution_status": execution_status,
                "message": "Scripts generated without execution"
            }
        
        # Save and execute each bash script
        if code_blocks:
            os.makedirs("generated_scripts", exist_ok=True)
            
            for i, code in enumerate(code_blocks):
                script_num = i
                logging.info(f"\n[SCRIPT {current_call}_{script_num:02d}] Preparing to execute on Kali Linux...")
                
                execution_count = 0
                last_error = None
                success = False
                script_paths = []

                while execution_count < max_depth:
                    try:
                        execution_count += 1
                        logging.info(f"\n[EXECUTION {current_call}_{script_num:02d}_{execution_count}] Attempting script execution...")
                        
                        # Save the script with the current execution number
                        script_content = code if code.strip().startswith('#!/') else f'#!/bin/bash\n\n{code}'
                        filename = f"generated_scripts/command_{current_call}_{script_num:02d}_{execution_count:02d}.sh"
                        with open(filename, "w") as f:
                            f.write(script_content.strip())
                        os.chmod(filename, 0o755)
                        script_paths.append(filename)
                        
                        # Let the agent decide whether and how to execute the script
                        agent_response = _agent.invoke(
                            {
                                "input": f"Execute the bash script number {script_num} that was just generated. " +
                                        f"The script contains the following code:\n{code}\n" +
                                        "If the script appears safe to execute, use the script_executor tool to run it. " +
                                        "If the execution is successful, respond with 'Final Answer' action."
                            }
                        )
                        
                        logging.info(f"\n[AGENT RESPONSE {current_call}_{script_num:02d}_{execution_count}] {agent_response}")
                        
                        # Check if script_executor was actually used and if it was successful
                        was_executed = "script_executor" in str(agent_response)
                        if was_executed:
                            # Check if the execution was successful
                            if "error" not in str(agent_response).lower() and "failed" not in str(agent_response).lower():
                                success = True
                                logging.info(f"[SCRIPT {current_call}_{script_num:02d}_{execution_count}] ✓ Successfully executed on Kali Linux")
                                break
                            else:
                                last_error = "Script execution failed, agent will retry if max_depth allows"
                        
                        if "Final Answer" in str(agent_response):
                            break
                            
                    except Exception as e:
                        last_error = str(e)
                        logging.error(f"[SCRIPT {current_call}_{script_num:02d}_{execution_count}] Error during execution: {last_error}")
                        if execution_count >= max_depth:
                            break
                
                # Record the final status for this script
                status = {
                    "script_number": script_num,
                    "call_number": current_call,
                    "executed_on_kali": was_executed,
                    "success": success,
                    "execution_attempts": execution_count,
                    "max_depth": max_depth,
                    "agent_response": agent_response,
                    "script_paths": script_paths
                }
                if last_error:
                    status["error"] = last_error
                execution_status.append(status)

        # Save script execution results to file
        with open(f"script_results_{current_call}.txt", "w") as f:
            for status in execution_status:
                script_num = status["script_number"]
                f.write(f"Call {current_call}, Script {script_num} results:\n")
                f.write(f"Script paths:\n")
                for path in status["script_paths"]:
                    f.write(f"  - {path}\n")
                f.write(f"Executed on Kali: {status['executed_on_kali']}\n")
                f.write(f"Success: {status['success']}\n")
                f.write(f"Execution attempts: {status['execution_attempts']}\n")
                f.write(f"Maximum allowed attempts: {status['max_depth']}\n")
                if status.get("error"):
                    f.write(f"Error:\n{status['error']}\n")
                f.write(f"Agent response:\n{status['agent_response']}\n\n")

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