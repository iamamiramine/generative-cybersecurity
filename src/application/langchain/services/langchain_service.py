import torch
import re
import os
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from src.application.system.services.system_service import execute_script

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.application.langchain.helpers.document_helper import split_documents, load_txt_files
from src.application.langchain.helpers.vector_db_helper import create_vector_db
from src.application.langchain.helpers.retreival_chain_helper import make_rag_chain, make_hyde_chain, prepare_hyde_prompt, prepare_rag_prompt, prepare_basic_prompt
from src.application.langchain.helpers.memory_helper import create_memory_chain

_model = None
_tokenizer = None
_hf = None

_ensemble_retriever = None
_retrieval_chain = None
_chain = None

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


def load_chain() -> dict:
    global _chain

    if not is_model_loaded():
        raise Exception("Model not loaded. Please call /load-model endpoint first")
    
    if _retrieval_chain == None:    
        prompt = prepare_basic_prompt(_tokenizer)
    else:
        prompt = None

    _chain = create_memory_chain(
        _hf, 
        _retrieval_chain, 
        StreamlitChatMessageHistory(key="langchain_messages"), 
        prompt,
    )  # Creates a memory chain using the loaded model, RAG chain, and Streamlit chat message history.
    return {"message": "Chain loaded successfully"}


def generate(question: str) -> str:
    torch.cuda.empty_cache()

    if not is_chain_loaded():
        raise Exception("Chain not loaded. Please call /load-chain endpoint first")

    output = _chain.invoke(        
        {"question": question},
        config={"configurable": {"session_id": "foo"}}
    )

    # Extract code blocks using regex
    code_blocks = re.findall(r'```(?:bash|sh)\n(.*?)\n```', output, re.DOTALL)
    
    # Save the complete response
    with open("output.txt", "w") as f:
        f.write(str(output))
    
    # Save each bash script
    if code_blocks:
        os.makedirs("generated_scripts", exist_ok=True)
        for i, code in enumerate(code_blocks):
            # Add shebang if not present
            if not code.strip().startswith('#!/'):
                code = '#!/bin/bash\n\n' + code
                
            filename = f"generated_scripts/command_{i+1}.sh"
            with open(filename, "w") as f:
                f.write(code.strip())
            # Make the script executable
            os.chmod(filename, 0o755)

    # Execute the generated scripts and collect results
    script_results = []
    for i in range(len(code_blocks)):
        result = execute_script(i + 1)
        script_results.append(result)

    # Save script execution results to file
    with open("script_results.txt", "w") as f:
        for i, result in enumerate(script_results, 1):
            f.write(f"Script {i} results:\n{result}\n\n")

    return output