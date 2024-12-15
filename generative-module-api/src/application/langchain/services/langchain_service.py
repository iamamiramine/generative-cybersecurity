import torch
import re
import os
from typing import Dict, List, Any
from langchain_huggingface.llms import HuggingFacePipeline

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.application.database.helpers.document_helper import save_text_chunks, load_text_chunks, has_dataset_changed, load_txt_files, split_documents
from src.application.database.helpers.vector_db_helper import create_vector_db
from src.application.langchain.helpers.retreival_chain_helper import make_rag_chain, make_hyde_chain, prepare_hyde_prompt, prepare_rag_prompt, prepare_basic_prompt
from src.application.langchain.helpers.memory_helper import create_memory_chain
from src.application.langchain.helpers.script_execution_tool_helper import execute_bash_script

from src.domain.models.langchain.langchain_models import LoadModelParameters, PipelineParameters, ContextRetrieverParameters, GenerateResponseParameters, ChainParameters
from src.application.langchain.models.langchain_state_model import LangchainState

state = LangchainState()


def load_model(load_model_parameters: LoadModelParameters) -> dict:    
    if state.is_model_loaded():
        return {"message": "Model already loaded"}
    
    try:
        if load_model_parameters.bit_quantization == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif load_model_parameters.bit_quantization == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported bit quantization: {load_model_parameters.bit_quantization}")
        
        state.tokenizer = AutoTokenizer.from_pretrained(
            load_model_parameters.model_path, 
            local_files_only=True, 
            trust_remote_code=True
        )
        state.model = AutoModelForCausalLM.from_pretrained(
            load_model_parameters.model_path, 
            torch_dtype=torch.float16,
            local_files_only=True, 
            device_map="auto",
            quantization_config=quantization_config, 
            trust_remote_code=False,
        )
        return {"message": "Model loaded successfully"}
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return {"error": str(e)}


def load_pipeline(pipeline_parameters: PipelineParameters) -> dict:    
    if not state.is_model_loaded():
        return {"error": "Model not loaded. Please call /load-model endpoint first"}
        
    try:
        task_type_str = pipeline_parameters.task_type.value
        if task_type_str not in ["text-generation", "text2text-generation"]:
            raise ValueError(f"Unsupported task type: {task_type_str}")
        
        # Debug logging
        print(f"Loading pipeline with model: {state.model}", flush=True)
        print(f"Loading pipeline with tokenizer: {state.tokenizer}", flush=True)
        
        state.task = task_type_str
        pipe = pipeline(
            task_type_str,  # Use the string directly
            model=state.model, 
            tokenizer=state.tokenizer, 
            max_new_tokens=pipeline_parameters.max_new_tokens,
            do_sample=pipeline_parameters.do_sample,
            temperature=pipeline_parameters.temperature,
            top_p=pipeline_parameters.top_p,
            top_k=pipeline_parameters.top_k,
            repetition_penalty=pipeline_parameters.repetition_penalty,
            pad_token_id=state.tokenizer.eos_token_id,
            eos_token_id=state.tokenizer.eos_token_id,
        )
        state.hf = HuggingFacePipeline(pipeline=pipe)
        return {"message": f"Pipeline loaded successfully for task: {task_type_str}"}
    except Exception as e:
        print(f"Error loading pipeline: {e}", flush=True)
        return {"error": str(e)}


def load_docs() -> dict:
    # TODO: Add check whether to load txt or pdf or csv
    try:
        state.docs = load_txt_files()
        return {"message": "Docs loaded successfully"}
    except Exception as e:
        print(f"Error loading docs: {e}", flush=True)
        return {"error": str(e)}


def load_ensemble_retriever_from_docs(force_reload: bool = False) -> dict:
    if not state.is_docs_loaded():
        return {"error": "Documents not loaded. Please call /load-docs endpoint first"}
    
    try:
        chunks_path = "./data/store/chunks.pkl"
        collection_name = "chroma"
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Check if dataset has changed
        dataset_changed = has_dataset_changed()
        should_reload = force_reload or dataset_changed
        
        if should_reload:
            print("Dataset changes detected or force reload requested", flush=True)
            texts = split_documents(state.docs)
            save_text_chunks(texts, chunks_path)
        else:
            if os.path.exists(chunks_path):
                print("Loading existing chunks (no dataset changes detected)", flush=True)
                texts = load_text_chunks(chunks_path)
            else:
                print("No existing chunks found, creating new ones", flush=True)
                texts = split_documents(state.docs)
                save_text_chunks(texts, chunks_path)
        
        # Create or load vector store
        vs = create_vector_db(
            texts=texts if should_reload else None,
            embeddings=embeddings,
            collection_name=collection_name,
            force_reload=should_reload
        )
        vs_retriever = vs.as_retriever()

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])
        
        # Create ensemble retriever
        state.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vs_retriever],
            weights=[0.5, 0.5]
        )
        
        return {"message": "Ensemble retriever loaded successfully"}
    except Exception as e:
        print(f"Error loading ensemble retriever: {e}", flush=True)
        return {"error": str(e)}


def load_context_retriever(context_retriever_parameters: ContextRetrieverParameters) -> dict:
    if not state.is_pipeline_loaded():
        return {"error": "Pipeline not loaded. Please call /load-pipeline endpoint first"}
    
    try:
        if context_retriever_parameters.use_hyde:
            hyde_generation_prompt, hyde_final_prompt = prepare_hyde_prompt(state.tokenizer)
            state.retrieval_chain = make_hyde_chain(
                state.hf, 
                state.ensemble_retriever, 
                hyde_generation_prompt, 
                hyde_final_prompt
            )
        else:
            rag_prompt = prepare_rag_prompt(state.tokenizer)
            state.retrieval_chain = make_rag_chain(
                state.hf, 
                state.ensemble_retriever, 
                rag_prompt
            )
        return {"message": "Retrieval chain loaded successfully"}
    except Exception as e:
        print(f"Error loading context retriever: {e}", flush=True)
        return {"error": str(e)}


def load_chain(chain_parameters: ChainParameters) -> dict:    
    if not state.is_pipeline_loaded():
        return {"error": "Pipeline not loaded. Please call /load-pipeline endpoint first"}
        
    try:
        if chain_parameters.chain_type == "basic":
            # basic_prompt = prepare_basic_prompt(state.tokenizer)
            state.chain = create_memory_chain(state.hf, None, state.messages)
        elif chain_parameters.chain_type in ["rag", "hyde"]:
            if not state.is_retriever_loaded():
                return {"error": "Retriever not loaded. Please call /load-retriever endpoint first"}
            
            if chain_parameters.chain_type == "rag":
                rag_prompt = prepare_rag_prompt(state.tokenizer)
                rag_chain = make_rag_chain(state.hf, state.ensemble_retriever, rag_prompt)
                state.chain = create_memory_chain(state.hf, rag_chain, state.messages)
            else:  # hyde
                hyde_generation_prompt, hyde_final_prompt = prepare_hyde_prompt(state.tokenizer)
                hyde_chain = make_hyde_chain(
                    state.hf, 
                    state.ensemble_retriever, 
                    hyde_generation_prompt, 
                    hyde_final_prompt
                )
                state.chain = create_memory_chain(state.hf, hyde_chain, state.messages)
        else:
            raise ValueError(f"Unsupported chain type: {chain_parameters.chain_type}")

        return {"message": f"{chain_parameters.chain_type} chain loaded successfully"}
    except Exception as e:
        print(f"Error loading chain: {e}", flush=True)
        return {"error": str(e)}


def generate(generate_response_parameters: GenerateResponseParameters) -> Dict[str, Any]:
    """Generate and optionally execute bash commands based on the user's question.
    
    Args:
        generate_response_parameters: The user's question/prompt and maximum number of execution attempts
    """
    if not state.is_chain_loaded():
        return {"error": "Chain not loaded. Please call /load-chain endpoint first"}
    
    try:
        state.call_counter += 1
        current_call = state.call_counter
        
        torch.cuda.empty_cache()
        print(f"\n[USER QUERY {current_call}] {generate_response_parameters.question}", flush=True)

        output = state.chain.invoke(        
            {"question": generate_response_parameters.question},
            config={"configurable": {"session_id": "foo"}}
        )
        print(f"\n[LLM RESPONSE {current_call}] {output}", flush=True)

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
                print(f"\n[SCRIPT {current_call}_{script_num:02d}] Processing script...", flush=True)
                
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
                if generate_response_parameters.max_depth > 0:
                    print(f"[SCRIPT {current_call}_{script_num:02d}] Executing on Kali Linux...", flush=True)
                    execution_result = execute_bash_script(script_path)
                    status.update({
                        "executed_on_kali": True,
                        "execution_result": execution_result,
                        "success": execution_result.get("success", False)
                    })
                    
                    if execution_result.get("success"):
                        print(f"[SCRIPT {current_call}_{script_num:02d}] ✓ Successfully executed", flush=True)
                    else:
                        error_msg = execution_result.get("error", "Unknown error")
                        print(f"[SCRIPT {current_call}_{script_num:02d}] ✗ Execution failed: {error_msg}", flush=True)
                else:
                    print(f"[SCRIPT {current_call}_{script_num:02d}] ℹ Script generated but not executed (max_depth=0)", flush=True)
                
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
        print(f"Error in generate function: {str(e)}", flush=True)  
        return {
            "error": True,
            "message": str(e),
            "details": "An error occurred during script generation or execution"
        }
    

def unload_components() -> dict:
    """Unload model, pipeline, and chain while preserving docs and retriever"""
    try:
        # Clear CUDA cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reset components in state
        state.model = None
        state.tokenizer = None
        state.hf = None
        state.chain = None
        state.task = None
        
        # Keep docs and retriever intact
        # state.docs and state.ensemble_retriever remain unchanged
        
        return {
            "message": "Components unloaded successfully",
            "status": {
                "model_loaded": state.is_model_loaded(),
                "pipeline_loaded": state.is_pipeline_loaded(),
                "docs_loaded": state.is_docs_loaded(),
                "retriever_loaded": state.is_retriever_loaded(),
                "chain_loaded": state.is_chain_loaded()
            }
        }
    except Exception as e:
        print(f"Error unloading components: {e}", flush=True)
        return {
            "error": str(e),
            "status": {
                "model_loaded": state.is_model_loaded(),
                "pipeline_loaded": state.is_pipeline_loaded(),
                "docs_loaded": state.is_docs_loaded(),
                "retriever_loaded": state.is_retriever_loaded(),
                "chain_loaded": state.is_chain_loaded()
            }
        }