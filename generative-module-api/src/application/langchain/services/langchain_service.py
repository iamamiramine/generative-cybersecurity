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
from src.application.database.helpers.vector_db_helper import create_vector_db, create_hyde_vector_store
from src.application.langchain.helpers.retreival_chain_helper import make_rag_chain, make_hyde_chain, prepare_hyde_prompt, prepare_rag_prompt, prepare_basic_prompt
from src.application.langchain.helpers.memory_helper import create_memory_chain
from src.application.langchain.helpers.script_execution_tool_helper import execute_bash_script

from src.domain.models.langchain.langchain_models import LoadModelParameters, PipelineParameters, GenerateResponseParameters, ChainParameters, EnsembleRetrieverParameters
from src.application.langchain.models.langchain_state_model import LangchainState

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder

state = LangchainState()


def load_model(load_model_parameters: LoadModelParameters) -> dict:    
    """
    Loads a language model with specified quantization parameters.
    
    Args:
        load_model_parameters: Parameters for loading the model including model path and bit quantization
        
    Returns:
        dict: A message indicating success or error
        
    The function:
    1. Checks if model is already loaded
    2. Configures quantization (4-bit or 8-bit)
    3. Loads tokenizer and model from local files
    4. Uses torch float16 precision and auto device mapping
    """
    # Check if model is already loaded to avoid duplicate loading
    if state.is_model_loaded():
        return {"message": "Model already loaded"}
    
    try:
        # Configure quantization based on bit parameter
        if load_model_parameters.bit_quantization == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif load_model_parameters.bit_quantization == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported bit quantization: {load_model_parameters.bit_quantization}")
        
        # Load tokenizer from local model files
        state.tokenizer = AutoTokenizer.from_pretrained(
            load_model_parameters.model_path, 
            local_files_only=True, 
            trust_remote_code=True
        )
        
        # Load model with quantization and device configurations
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
    # Check if model is loaded before proceeding
    if not state.is_model_loaded():
        return {"error": "Model not loaded. Please call /load-model endpoint first"}
        
    try:
        # Get task type from parameters and validate it
        task_type_str = pipeline_parameters.task_type.value
        if task_type_str not in ["text-generation", "text2text-generation"]:
            raise ValueError(f"Unsupported task type: {task_type_str}")
        
        # Log pipeline loading details
        print(f"Loading pipeline with model: {state.model}", flush=True)
        print(f"Loading pipeline with tokenizer: {state.tokenizer}", flush=True)
        
        # Store task type in state
        state.task = task_type_str

        # Create HuggingFace pipeline with specified parameters
        pipe = pipeline(
            task_type_str,
            model=state.model, 
            tokenizer=state.tokenizer, 
            max_new_tokens=pipeline_parameters.max_new_tokens,  # Maximum number of tokens to generate
            do_sample=pipeline_parameters.do_sample,           # Whether to use sampling
            temperature=pipeline_parameters.temperature,       # Controls randomness in sampling
            top_p=pipeline_parameters.top_p,                  # Nucleus sampling parameter
            top_k=pipeline_parameters.top_k,                  # Top-k sampling parameter
            repetition_penalty=pipeline_parameters.repetition_penalty,  # Penalize repeated tokens
            pad_token_id=state.tokenizer.eos_token_id,       # Use EOS token as padding
            eos_token_id=state.tokenizer.eos_token_id,       # End of sequence token
        )

        # Create LangChain pipeline wrapper
        state.hf = HuggingFacePipeline(pipeline=pipe)
        return {"message": f"Pipeline loaded successfully for task: {task_type_str}"}
    except Exception as e:
        # Log and return any errors that occur
        print(f"Error loading pipeline: {e}", flush=True)
        return {"error": str(e)}


def load_docs() -> dict:
    """
    Load documents into the application state.
    Currently only supports loading text files, but future support planned for PDF and CSV.
    
    Returns:
        dict: A dictionary containing either:
            - {"message": "Docs loaded successfully"} on success
            - {"error": <error message>} on failure
    """
    # TODO: Add check whether to load txt or pdf or csv
    try:
        # Load text files into application state
        state.docs = load_txt_files()
        
        # Return success message
        return {"message": "Docs loaded successfully"}
    except Exception as e:
        # Log error and return error message if loading fails
        print(f"Error loading docs: {e}", flush=True)
        return {"error": str(e)}


def load_ensemble_retriever_from_docs(ensemble_retriever_parameters: EnsembleRetrieverParameters) -> dict:
    """
    Load an ensemble retriever that combines BM25 and vector similarity search.
    
    Args:
        ensemble_retriever_parameters: Parameters for configuring the ensemble retriever
        
    Returns:
        dict: Success/error message
    """
    # Check if documents are loaded before proceeding
    if not state.is_docs_loaded():
        return {"error": "Documents not loaded. Please call /load-docs endpoint first"}
    
    try:
        # Define paths and configuration
        chunks_path = "./data/store/chunks.pkl"
        collection_name = "chroma"
        # Initialize sentence embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Check if we need to reload the dataset
        dataset_changed = has_dataset_changed()
        should_reload = ensemble_retriever_parameters.force_reload or dataset_changed
        
        # Handle document chunking based on reload conditions
        if should_reload:
            print("Dataset changes detected or force reload requested", flush=True)
            texts = split_documents(state.docs)  # Split docs into chunks
            save_text_chunks(texts, chunks_path) # Cache chunks
        else:
            if os.path.exists(chunks_path):
                print("Loading existing chunks (no dataset changes detected)", flush=True)
                texts = load_text_chunks(chunks_path)
            else:
                print("No existing chunks found, creating new ones", flush=True)
                texts = split_documents(state.docs)
                save_text_chunks(texts, chunks_path)
        
        # Handle HyDE (Hypothetical Document Embeddings) if enabled
        if ensemble_retriever_parameters.use_hyde:
            # Create HyDE embedder using the loaded language model
            hyde = HypotheticalDocumentEmbedder.from_llm(
                llm=state.hf,
                embeddings=embeddings,
                prompt_template=prepare_hyde_prompt(state.tokenizer)[0]  # Use generation prompt
            )
            # Generate hypothetical document embeddings
            hypothetical_embeddings = hyde.embed_documents(texts)
            
            # Create vector store with HyDE embeddings
            vs = create_hyde_vector_store(
                texts=texts,
                embeddings=embeddings,
                hypothetical_embeddings=hypothetical_embeddings,
                collection_name=collection_name
            )
        else:
            # Create standard vector store without HyDE
            vs = create_vector_db(
                texts=texts if should_reload else None,
                embeddings=embeddings,
                collection_name=collection_name,
                force_reload=should_reload
            )
        # Convert vector store to retriever
        vs_retriever = vs.as_retriever()

        # Create BM25 retriever from document texts
        bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])
        
        # Combine both retrievers with equal weights
        state.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vs_retriever],
            weights=[0.5, 0.5]
        )
        
        return {"message": "Ensemble retriever loaded successfully"}
    except Exception as e:
        print(f"Error loading ensemble retriever: {e}", flush=True)
        return {"error": str(e)}


def load_chain(chain_parameters: ChainParameters) -> dict:    
    """
    Loads and configures a language model chain based on the specified parameters.
    
    Args:
        chain_parameters: Configuration parameters for the chain type and behavior
        
    Returns:
        dict: Status message indicating success or error
        
    The function supports three types of chains:
    - basic: Simple chain without retrieval
    - rag: Chain with Retrieval Augmented Generation
    - hyde: Chain with Hypothetical Document Embeddings
    """
    
    # Verify pipeline is loaded before proceeding
    if not state.is_pipeline_loaded():
        return {"error": "Pipeline not loaded. Please call /load-pipeline endpoint first"}
        
    try:
        if chain_parameters.chain_type == "basic":
            # Create basic chain without retrieval capabilities
            # basic_prompt = prepare_basic_prompt(state.tokenizer)
            state.chain = create_memory_chain(state.hf, None, state.messages)
            
        elif chain_parameters.chain_type in ["rag", "hyde"]:
            # Verify retriever is loaded for RAG and HyDE chains
            if not state.is_retriever_loaded():
                return {"error": "Retriever not loaded. Please call /load-retriever endpoint first"}
            
            if chain_parameters.chain_type == "rag":
                # Configure RAG chain with retrieval capabilities
                rag_prompt = prepare_rag_prompt(state.tokenizer)
                rag_chain = make_rag_chain(state.hf, state.ensemble_retriever, rag_prompt)
                state.chain = create_memory_chain(state.hf, rag_chain, state.messages)
            else:  # hyde
                # Configure HyDE chain with hypothetical document embeddings
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
        # Log and return any errors that occur during chain loading
        print(f"Error loading chain: {e}", flush=True)
        return {"error": str(e)}


def generate(generate_response_parameters: GenerateResponseParameters) -> Dict[str, Any]:
    """
    Generate and optionally execute scripts based on user input using the loaded chain.
    
    Args:
        generate_response_parameters: Parameters containing the question and execution settings
        
    Returns:
        Dict containing LLM output, call tracking info, and script execution results
    """
    # Verify chain is loaded before proceeding
    if not state.is_chain_loaded():
        return {"error": "Chain not loaded. Please call /load-chain endpoint first"}
    
    try:
        # Increment and store call counter for tracking multiple generations
        state.call_counter += 1
        current_call = state.call_counter
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        print(f"\n[USER QUERY {current_call}] {generate_response_parameters.question}", flush=True)

        # Generate response from chain
        output = state.chain.invoke(        
            {"question": generate_response_parameters.question},
            config={"configurable": {"session_id": "foo"}}
        )
        print(f"\n[LLM RESPONSE {current_call}] {output}", flush=True)

        # Extract bash code blocks from response
        code_blocks = re.findall(r'```(?:bash|sh)\n(.*?)\n```', output, re.DOTALL)
        
        # Save raw output to file
        with open(f"output_{current_call}.txt", "w") as f:
            f.write(str(output))
        
        # Track execution status of each script
        execution_status: List[Dict[str, Any]] = []
        
        if code_blocks:
            # Create directory for generated scripts
            os.makedirs("generated_scripts", exist_ok=True)
            
            # Process each code block
            for i, code in enumerate(code_blocks):
                script_num = i
                print(f"\n[SCRIPT {current_call}_{script_num:02d}] Processing script...", flush=True)
                
                # Add shebang if missing and save script
                script_content = code if code.strip().startswith('#!/') else f'#!/bin/bash\n\n{code}'
                script_path = f"generated_scripts/command_{current_call}_{script_num:02d}.sh"
                
                with open(script_path, "w") as f:
                    f.write(script_content.strip())
                os.chmod(script_path, 0o755)  # Make script executable
                
                # Initialize status tracking for this script
                status = {
                    "script_number": script_num,
                    "call_number": current_call,
                    "script_content": script_content,
                    "script_path": script_path,
                    "executed_on_kali": False,
                    "execution_result": None
                }
                
                # Execute script if max_depth allows
                if generate_response_parameters.max_depth > 0:
                    print(f"[SCRIPT {current_call}_{script_num:02d}] Executing on Kali Linux...", flush=True)
                    execution_result = execute_bash_script(script_path)
                    status.update({
                        "executed_on_kali": True,
                        "execution_result": execution_result,
                        "success": execution_result.get("success", False)
                    })
                    
                    # Log execution status
                    if execution_result.get("success"):
                        print(f"[SCRIPT {current_call}_{script_num:02d}] ✓ Successfully executed", flush=True)
                    else:
                        error_msg = execution_result.get("error", "Unknown error")
                        print(f"[SCRIPT {current_call}_{script_num:02d}] ✗ Execution failed: {error_msg}", flush=True)
                else:
                    print(f"[SCRIPT {current_call}_{script_num:02d}] ℹ Script generated but not executed (max_depth=0)", flush=True)
                
                execution_status.append(status)

        # Save detailed execution results to file
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

        # Return generation and execution results
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
