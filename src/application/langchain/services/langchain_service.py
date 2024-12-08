import torch
import re
import os
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from src.application.langchain.services.retriever_service import load_retriever
from src.application.system.services.system_service import execute_script
from src.application.langchain.services.ragchain_service import make_rag_chain, make_hyde_chain
from src.application.langchain.services.memory_service import create_memory_chain


_model = None
_tokenizer = None
_hf = None

_retriever = None
_rag_chain = None
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
        max_new_tokens=4000,
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


def load_chain(use_hyde: bool = False) -> dict:
    global _chain
    global _retriever
    global _rag_chain

    if not is_model_loaded():
        raise Exception("Model not loaded. Please call /load-model endpoint first")

    # system = """You are an AI that writes bash scripts. Please answer with bash scripts only, and make sure to format with codeblocks using ```bash and ```."""

    # user_input = """{question}"""

    # conversation = """Bash Script:"""

    # chat = [
    #     {"role": "system", "content": system},
    #     {"role": "user", "content": user_input},
    #     {"role": "assistant", "content": conversation},
    # ]

    # prompt = _tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # system = """You are an AI that writes bash scripts. Using the following context, please answer with bash scripts only, and make sure to format with codeblocks using ```bash and ```."""
    _retriever = load_retriever()

    if use_hyde:
        system = """You are an expert in bash scripting. Given a question about bash commands, please provide an answer with bash scripts only, and make sure to format with codeblocks using ```bash and ```."""

        hyde_generation_template = """Given the following question, generate a detailed hypothetical document that would contain the answer to the question.
        
        Question: {question}
        
        Hypothetical Document:"""
        hyde_generation_chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": hyde_generation_template},
        ]

        hyde_generation_prompt = _tokenizer.apply_chat_template(hyde_generation_chat, tokenize=False, add_generation_prompt=True)
        
        # Second prompt for final answer using hypothetical doc and retrieved context
        final_prompt_template = """You are an expert in bash scripting. Given a question about bash commands, please provide an answer with bash scripts only, and make sure to format with codeblocks using ```bash and ```.
        Hypothetical Document: {hypothetical_document}
        Retrieved Context: {context}
        
        Question: """

        final_chat = [
            {"role": "system", "content": final_prompt_template},
            {"role": "user", "content": "{question}"},
        ]
        final_prompt = _tokenizer.apply_chat_template(final_chat, tokenize=False, add_generation_prompt=True)

        _retrieval_chain = make_hyde_chain(_hf, _retriever, hyde_generation_prompt, final_prompt)
        
    else:
        system = """You are an expert in bash scripting. Given a question about bash commands, please provide an answer with bash scripts only, and make sure to format with codeblocks using ```bash and ```.
        
        Context: {context}
        
        Question: """

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        # chat = [
        #     {"role": "system", "content": system},
        #     {"role": "user", "content": "{question}"},
        # ]

        # rag_prompt = _tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        _retrieval_chain = make_rag_chain(_hf, _retriever, rag_prompt)
    
    _chain = create_memory_chain(_hf, _retrieval_chain, StreamlitChatMessageHistory(key="langchain_messages"))  # Creates a memory chain using the loaded model, RAG chain, and Streamlit chat message history.
    return {"message": "Chain loaded successfully"}


def generate(question: str) -> str:
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

    print(script_results, flush=True)

    return output