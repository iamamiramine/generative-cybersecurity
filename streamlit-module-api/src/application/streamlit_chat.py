import os
import json
from typing import Dict

import streamlit as st
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
@st.cache_data
def get_config():
    """Load API configuration from JSON file and return complete config"""
    try:
        # Assuming the config file is in a relative path from this file
        config_path = os.path.join("shared", "config", "api_config.json")
        
        with open(config_path) as f:
            api_config = json.load(f)
            
        return {
            "BASE_URL": api_config["generative_module"],
            "LANGCHAIN_URL": f"{api_config['generative_module']}/langchain",
            # "LOCAL_LLM_URL": f"{api_config["generative_module"]}/local_llm",
            "ENDPOINTS": {
                "model": "/load_model",
                "pipeline": "/load_pipeline",
                "docs": "/load_docs",
                "retriever": "/load_ensemble_retriever_from_docs",
                "chain": "/load_chain",
                "generate": "/generate"
            }
        }
    
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        st.error("Failed to load API configuration")
        return None
    

def check_service_status() -> Dict[str, bool]:
    """Check the current status of the LangChain service"""
    config = get_config()
    try:
        response = requests.get(f"{config['LANGCHAIN_URL']}/status")
        if response.ok:
            return response.json()
        return {
            "model_loaded": False,
            "pipeline_loaded": False,
            "docs_loaded": False,
            "retriever_loaded": False,
            "chain_loaded": False
        }
    except Exception as e:
        logger.error(f"Error checking service status: {e}")
        return {
            "model_loaded": False,
            "pipeline_loaded": False,
            "docs_loaded": False,
            "retriever_loaded": False,
            "chain_loaded": False
        }

 
# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_context" not in st.session_state:
        st.session_state.current_context = None
    # Add default generation parameters
    if "generation_params" not in st.session_state:
        st.session_state.generation_params = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        }
    
    # Debug config loading
    if st.config.get_option("server.headless"):
        logging.info("Running in headless mode")
    if st.config.get_option("logger.level"):
        logging.info(f"Logger level: {st.config.get_option('logger.level')}")


# UI Components
def render_header():
    logo_path = os.path.join("shared", "assets", "Logo.png")
    st.image(logo_path, width=100)
    st.title("Cyber Rabbit")


def render_generation_controls():
    """Render controls for generation parameters"""
    with st.sidebar:
        st.subheader("Generation Settings")
        
        # Pipeline parameters
        st.session_state.generation_params["temperature"] = st.slider(
            "Temperature", 0.0, 2.0, 
            st.session_state.generation_params["temperature"], 
            help="Higher values make output more random, lower values more deterministic"
        )
        
        st.session_state.generation_params["top_p"] = st.slider(
            "Top P", 0.0, 1.0, 
            st.session_state.generation_params["top_p"],
            help="Nucleus sampling: limits cumulative probability of tokens considered"
        )
        
        st.session_state.generation_params["top_k"] = st.slider(
            "Top K", 1, 100, 
            st.session_state.generation_params["top_k"],
            help="Limits the number of tokens considered for each step"
        )
        
        st.session_state.generation_params["max_new_tokens"] = st.slider(
            "Max New Tokens", 32, 1024, 
            st.session_state.generation_params["max_new_tokens"],
            help="Maximum length of generated response"
        )
        
        st.session_state.generation_params["repetition_penalty"] = st.slider(
            "Repetition Penalty", 1.0, 2.0, 
            st.session_state.generation_params["repetition_penalty"],
            help="Higher values reduce repetition in generated text"
        )
        
        # Chain type selection (only show when context is "File")
        if st.session_state.current_context == "File":
            chain_type = st.radio(
                "Chain Type",
                ["rag", "hyde"],
                help="RAG: Regular retrieval, HYDE: Hypothetical document embeddings"
            )
            st.session_state.generation_params["chain_type"] = chain_type



@st.cache_data
def load_context(context: str) -> bool:
    """Load necessary models and data for the selected context"""
    config = get_config()
    if not config:
        return False

    try:
        # Check current status
        status = check_service_status()
        
        # Step 1: Load model if not loaded
        if not status["model_loaded"]:
            model_params = {
                "model_path": "models/WhiteRabbitNeo_WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B",
                "bit_quantization": 4
            }
            response = requests.post(
                f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['model']}", 
                json=model_params
            )
            if not response.ok:
                st.error(f"Failed to load model: {response.json().get('detail', 'Unknown error')}")
                return False

        # Step 2: Load pipeline with user parameters
        if not status["pipeline_loaded"]:
            pipeline_params = {
                "max_new_tokens": st.session_state.generation_params["max_new_tokens"],
                "do_sample": st.session_state.generation_params["do_sample"],
                "temperature": st.session_state.generation_params["temperature"],
                "top_p": st.session_state.generation_params["top_p"],
                "top_k": st.session_state.generation_params["top_k"],
                "repetition_penalty": st.session_state.generation_params["repetition_penalty"]
            }
            response = requests.post(
                f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['pipeline']}", 
                json=pipeline_params
            )
            if not response.ok:
                st.error(f"Failed to load pipeline: {response.json().get('detail', 'Unknown error')}")
                return False

        # Context-specific loading
        if context == "File":
            # Step 3: Load documents if not loaded
            if not status["docs_loaded"]:
                response = requests.post(f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['docs']}")
                if not response.ok:
                    st.error(f"Failed to load documents: {response.json().get('detail', 'Unknown error')}")
                    return False
            
            # Step 4: Load ensemble retriever if not loaded
            if not status["retriever_loaded"]:
                response = requests.post(f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['retriever']}")
                if not response.ok:
                    st.error(f"Failed to load retriever: {response.json().get('detail', 'Unknown error')}")
                    return False
            
            # Step 5: Load RAG chain
            chain_params = {
                "chain_type": st.session_state.generation_params.get("chain_type", "rag")
            }
        else:
            # Load basic chain for non-File context
            chain_params = {
                "chain_type": "basic"
            }
        
        # Step 6: Load chain if not loaded or if chain type changed
        response = requests.post(
            f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['chain']}", 
            json=chain_params
        )
        if not response.ok:
            st.error(f"Failed to load chain: {response.json().get('detail', 'Unknown error')}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading context: {e}")
        st.error(f"Error loading context: {str(e)}")
        return False


def handle_context_switch(context: str):
    """Handle context switching with proper feedback"""
    if st.session_state.current_context != context:
        with st.spinner(f"Switching context to '{context}' and reloading..."):
            # Check current status before loading
            status = check_service_status()
            if not status["model_loaded"]:
                st.warning("Model not loaded. Loading required components...")
            
            success = load_context(context)
            if success:
                st.session_state.current_context = context
                st.success(f"Switched to '{context}' context successfully!")
            else:
                st.error("Failed to switch context. Please try again.")


def generate_response(prompt: str) -> str:
    """Generate response from the API with error handling"""
    config = get_config()
    if not config:
        return "Error: Unable to load API configuration"
    
    try:
        # Check if chain is loaded before generating
        status = check_service_status()
        if not status["chain_loaded"]:
            return "Error: Chain not loaded. Please switch context first."

        # Generate response with parameters
        generate_params = {
            "question": prompt,
            "max_depth": 1
        }
        response = requests.post(
            f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['generate']}", 
            json=generate_params
        )
        
        if not response.ok:
            error_detail = response.json().get('detail', 'Unknown error')
            logger.error(f"Error from API: {error_detail}")
            return f"Error generating response: {error_detail}"

        response_data = response.json()
        if "llm_output" in response_data:
            return response_data["llm_output"]
        return response_data.get("response", "Sorry, I couldn't generate a response.")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"


def display_chat_message(message: str, stream: bool = True):
    """Display chat message with optional streaming effect"""
    if stream:
        chat_placeholder = st.empty()
        for i in range(len(message)):
            chat_placeholder.markdown(message[:i+1], unsafe_allow_html=True)
            time.sleep(0.05)
    else:
        st.markdown(message)


def main():
    init_session_state()
    render_header()
    render_generation_controls() 
    
    # Context selector
    context = st.selectbox("Select Context", ["None", "File"], index=0)
    handle_context_switch(context)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            display_chat_message(message["content"], stream=False)

    # Chat input and response
    if prompt := st.chat_input("Ask me anything!"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display response
        with st.spinner("Generating response..."):
            reply = generate_response(prompt)
            with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                display_chat_message(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()