import os
import json
import re

import streamlit as st
import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@st.cache_data  # Cache the config to avoid reloading on every rerun
def get_config():
    """Load API configuration from JSON file and return complete config"""
    try:
        # Construct path to config file in shared/config directory
        config_path = os.path.join("shared", "config", "api_config.json")

        # Load and parse JSON config file
        with open(config_path) as f:
            api_config = json.load(f)

        # Return dictionary with API configuration    
        return {
            # Base URL for the generative module API
            "BASE_URL": api_config["generative_module"],
            # URL for LangChain specific endpoints
            "LANGCHAIN_URL": f"{api_config['generative_module']}/langchain",
            # Dictionary of API endpoint paths
            "ENDPOINTS": {
                "model": "/load_model",         # Endpoint to load ML model
                "pipeline": "/load_pipeline",   # Endpoint to load generation pipeline
                "docs": "/load_docs",           # Endpoint to load documents
                "retriever": "/load_ensemble_retriever_from_docs",  # Load document retriever
                "chain": "/load_chain",         # Load LangChain chain
                "generate": "/generate"         # Generate text endpoint
            }
        }

    except Exception as e:
        # Log error and show error message in Streamlit UI
        logger.error(f"Error loading config: {e}")
        st.error("Failed to load API configuration") 
        return None
 
def init_session_state():
    """Initialize Streamlit session state variables and default parameters"""
    # Initialize chat message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize current context for RAG
    if "current_context" not in st.session_state:
        st.session_state.current_context = None

    # Initialize SSH session state
    if "ssh_session" not in st.session_state:
        st.session_state.ssh_session = {
            "active": False,
            "host": None,
            "user": None,
            "control_path": None,
            "last_command_output": None
        }

    # Default parameters for text generation pipeline
    default_params = {
        "max_new_tokens": 2048,    # Maximum number of NEW tokens to generate (excluding prompt)
        "do_sample": True,        # Enable sampling (vs greedy decoding)
        "temperature": 0.7,       # Controls randomness (higher = more random)
        "top_p": 0.9,            # Nucleus sampling parameter
        "top_k": 50,             # Top-k sampling parameter
        "repetition_penalty": 1.1,# Penalize repeated tokens
        "max_depth": 1,          # Defines whether the script would be run or not on the Kali machine
        "chain_type": "rag"      # Type of LangChain to use (RAG = Retrieval Augmented Generation)
    }

    # Initialize pipeline parameters if not already set
    if "pipeline_params" not in st.session_state:
        st.session_state.pipeline_params = default_params.copy()
    
    # Keep track of previous pipeline parameters to detect changes
    if "previous_pipeline_params" not in st.session_state:
        st.session_state.previous_pipeline_params = default_params.copy()
    
    # Track model and document loading status
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    
    # Log server configuration details
    if st.config.get_option("server.headless"):
        logging.info("Running in headless mode")
    if st.config.get_option("logger.level"):
        logging.info(f"Logger level: {st.config.get_option('logger.level')}")


@st.cache_data  # Cache the header rendering to improve performance
def render_header():
    """Renders the application header with logo and title"""
    # Construct path to logo image file
    logo_path = os.path.join("shared", "assets", "Logo.png")
    
    # Display logo image with width of 100 pixels
    st.image(logo_path, width=100)
    
    # Display application title
    st.title("Cyber Rabbit")


def render_generation_controls():
    """Render controls for generation parameters in the sidebar
    
    This function creates sliders and radio buttons in the Streamlit sidebar
    to control various text generation parameters including:
    - Temperature: Controls randomness of output
    - Top P: Controls nucleus sampling threshold
    - Top K: Controls number of tokens considered
    - Max New Tokens: Controls length of generated text
    - Repetition Penalty: Controls repetition in output
    - Max Depth: Controls context retrieval depth
    - Chain Type: Selects between RAG and HYDE approaches
    """
    with st.sidebar:
        # Add a subheader for the generation settings section
        st.subheader("Generation Settings")
        
        # Temperature slider - controls randomness vs determinism
        # Higher values (>1.0) increase randomness, lower values (<1.0) make output more focused
        st.session_state.pipeline_params["temperature"] = st.slider(
            "Temperature", 0.0, 2.0, 
            st.session_state.pipeline_params["temperature"],
            help="Higher values make output more random, lower values more deterministic"
        )
        
        # Top P slider - implements nucleus sampling
        # Only tokens with cumulative probability < top_p are considered
        st.session_state.pipeline_params["top_p"] = st.slider(
            "Top P", 0.0, 1.0, 
            st.session_state.pipeline_params["top_p"],
            help="Nucleus sampling: limits cumulative probability of tokens considered"
        )
        
        # Top K slider - limits token consideration pool
        # Only the top k most likely tokens are considered for sampling
        st.session_state.pipeline_params["top_k"] = st.slider(
            "Top K", 1, 100, 
            st.session_state.pipeline_params["top_k"],
            help="Limits the number of tokens considered for each step"
        )
        
        # Max New Tokens slider - controls response length
        # Sets upper limit on number of tokens in generated response
        st.session_state.pipeline_params["max_new_tokens"] = st.slider(
            "Max New Tokens", 256, 4096, 
            st.session_state.pipeline_params["max_new_tokens"],
            help="Maximum length of generated response (excluding prompt)"
        )
        
        # Repetition Penalty slider - controls token reuse
        # Higher values (>1.0) make the model less likely to repeat tokens
        st.session_state.pipeline_params["repetition_penalty"] = st.slider(
            "Repetition Penalty", 1.0, 2.0, 
            st.session_state.pipeline_params["repetition_penalty"],
            help="Higher values reduce repetition in generated text"
        )

        # Max Depth slider - controls context retrieval depth
        # Higher values allow more context to be retrieved but may slow performance
        st.session_state.pipeline_params["max_depth"] = st.slider(
            "Max Depth", 0, 1, 
            st.session_state.pipeline_params.get("max_depth", 1),
            help="Maximum depth of the command execution"
        )
        
        # Chain Type selector - chooses between RAG and HYDE approaches
        # RAG uses standard retrieval, HYDE uses hypothetical document embeddings
        chain_type = st.radio(
            "Chain Type",
            ["rag", "hyde"],
            help="RAG: Regular retrieval, HYDE: Hypothetical document embeddings"
        )
        st.session_state.pipeline_params["chain_type"] = chain_type


def initialize_base_components():
    """Initialize documents and retriever at startup"""
    # Check if components are already loaded to avoid reloading
    if st.session_state.model_loaded and st.session_state.docs_loaded:
        return True

    # Get configuration settings
    config = get_config()
    if not config:
        return False

    try:
        # Load documents if not already loaded
        if not st.session_state.docs_loaded:
            st.info("Loading documents...")
            # Make API call to load documents
            response = requests.post(f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['docs']}")
            if not response.ok:
                st.error(f"Failed to load documents: {response.json().get('detail', 'Unknown error')}")
                return False
            st.session_state.docs_loaded = True
            st.success("Documents loaded successfully!")

        # Load AI model if not already loaded
        if not st.session_state.model_loaded:
            # Set model parameters for initialization
            model_params = {
                "model_path": "models/WhiteRabbitNeo_WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B",
                "bit_quantization": 4  # Use 4-bit quantization for efficiency
            }
            # Make API call to load model
            response = requests.post(
                f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['model']}", 
                json=model_params
            )
            if not response.ok:
                st.error(f"Failed to load model: {response.json().get('detail', 'Unknown error')}")
                return False
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")

        # Initialize the pipeline with current parameters
        st.info("Loading pipeline...")
        response = requests.post(
            f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['pipeline']}", 
            json=st.session_state.pipeline_params
        )
        if not response.ok:
            st.error(f"Failed to load pipeline: {response.json().get('detail', 'Unknown error')}")
            return False
        st.success("Pipeline loaded successfully!")

        return True

    except Exception as e:
        # Log any errors that occur during initialization
        logger.error(f"Error loading base components: {e}")
        st.error(f"Error loading base components: {str(e)}")
        return False


def load_pipeline(pipeline_params: dict) -> bool:
    """Load pipeline with current generation parameters"""
    # Get configuration settings
    config = get_config()
    if not config:
        return False

    try:
        # Make API call to load pipeline with provided parameters
        response = requests.post(
            f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['pipeline']}", 
            json=pipeline_params
        )
        
        # Check if API call was successful
        if not response.ok:
            # Display error message if pipeline loading failed
            st.error(f"Failed to load pipeline: {response.json().get('detail', 'Unknown error')}")
            return False
            
        # Display success message
        st.success("Pipeline loaded successfully!")
            
        return True

    except Exception as e:
        # Log error and display message if exception occurs
        logger.error(f"Error loading pipeline: {e}")
        st.error(f"Error loading pipeline: {str(e)}")
        return False
    

def load_context(context: str) -> bool:
    """Load context-specific chain"""
    # Get configuration settings
    config = get_config()
    if not config:
        return False

    try:
        # Determine chain type based on context
        # For File context, get chain_type from pipeline params (default to "rag")
        # For other contexts, use "basic" chain type
        if context == "File":
            chain_type = st.session_state.pipeline_params.get("chain_type", "rag")
        else:
            chain_type = "basic"
            st.session_state.pipeline_params["chain_type"] = chain_type
        
        # Prepare parameters for chain initialization
        chain_params = {"chain_type": chain_type}
        # HyDE (Hypothetical Document Embeddings) is only used for specific chain type
        use_hyde = chain_type == "hyde"

        # First load the retriever component
        st.info("Loading retriever...")
        response = requests.post(f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['retriever']}", json={"use_hyde": use_hyde})
        if not response.ok:
            st.error(f"Failed to load retriever: {response.json().get('detail', 'Unknown error')}")
            return False
        st.success("Retriever loaded successfully!")
        
        # Then initialize the chain with specified parameters
        response = requests.post(
            f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['chain']}", 
            json=chain_params
        )
        if not response.ok:
            st.error(f"Failed to load chain: {response.json().get('detail', 'Unknown error')}")
            return False
        st.success("Chain loaded successfully!")
        
        return True
        
    except Exception as e:
        # Log any errors that occur during context loading
        logger.error(f"Error loading context: {e}")
        st.error(f"Error loading context: {str(e)}")
        return False
    

def handle_pipeline_switch():
    """Handle pipeline switching with proper feedback"""
    # Create dictionary of current pipeline parameters
    current_params = {
        # Maximum number of tokens to generate in response
        "max_new_tokens": st.session_state.pipeline_params["max_new_tokens"],
        # Whether to use sampling in text generation
        "do_sample": st.session_state.pipeline_params["do_sample"], 
        # Controls randomness in generation (higher = more random)
        "temperature": st.session_state.pipeline_params["temperature"],
        # Nucleus sampling parameter (higher = more diverse)
        "top_p": st.session_state.pipeline_params["top_p"],
        # Top-k sampling parameter (higher = more options)
        "top_k": st.session_state.pipeline_params["top_k"],
        # Penalty for repeating tokens (higher = less repetition)
        "repetition_penalty": st.session_state.pipeline_params["repetition_penalty"],
        # Maximum recursion depth for chain operations
        "max_depth": st.session_state.pipeline_params.get("max_depth", 1),
        # Type of chain to use (e.g. "rag" for retrieval-augmented generation)
        "chain_type": st.session_state.pipeline_params.get("chain_type", "rag")
    }
        
    # Check if parameters have changed from previous state
    if current_params != st.session_state.previous_pipeline_params:
        # Show loading spinner while switching pipeline
        with st.spinner("Switching pipeline and reloading..."):
            # Attempt to load new pipeline with current parameters
            success = load_pipeline(current_params)
            if success:
                # Update previous parameters and show success message
                st.session_state.previous_pipeline_params = current_params.copy()
                st.success("Switched to pipeline successfully!")
            else:
                # Show error message if pipeline switch fails
                st.error("Failed to switch pipeline. Please try again.")


def handle_context_switch(context: str):
    """Handle context switching with proper feedback"""
    # Check if the requested context is different from current context
    if st.session_state.current_context != context:
        # Show loading spinner while context switch is in progress
        with st.spinner(f"Switching context to '{context}' and reloading..."):
            # Update the current context in session state
            st.session_state.current_context = context
            # Attempt to load the new context
            success = load_context(context)
            
            # Show success message if context switch succeeded
            if success:
                st.success(f"Switched to '{context}' context successfully!")
            # Show error message if context switch failed    
            else:
                st.error("Failed to switch context. Please try again.")


def generate_response(prompt: str) -> str:
    """
    Generate a response to a given prompt using the configured API endpoint.
    
    Args:
        prompt (str): The user's input prompt/question
        
    Returns:
        dict: The response data containing LLM output and execution information
    """
    # Get API configuration settings
    config = get_config()
    if not config:
        return {"error": "Unable to load API configuration"}
    
    try:
        # Prepare parameters for the generate API endpoint
        generate_params = {
            "question": prompt,
            "max_depth": st.session_state.pipeline_params.get("max_depth", 1)
        }
        
        # Make POST request to generate endpoint
        response = requests.post(
            f"{config['LANGCHAIN_URL']}{config['ENDPOINTS']['generate']}", 
            json=generate_params
        )

        # Handle unsuccessful API responses
        if not response.ok:
            error_detail = response.json().get('detail', 'Unknown error')
            logger.error(f"Error from API: {error_detail}")
            return {"error": f"Error generating response: {error_detail}"}

        # Return the raw response data
        return response.json()
        
    except Exception as e:
        # Log and return any unexpected errors
        logger.error(f"Error generating response: {e}")
        return {"error": f"Error generating response: {str(e)}"}


def display_chat_message(message: dict):
    """Display chat message with proper formatting
    
    Args:
        message (dict): Message dictionary containing role, content, and format
    """
    if message.get("format") == "code":
        st.code(message["content"], language="bash")
    else:
        st.markdown(message["content"], unsafe_allow_html=True)


def update_ssh_session_state(script_content: str, execution_result: dict):
    """Update SSH session state based on script execution
    
    Args:
        script_content (str): The content of the executed script
        execution_result (dict): The result of script execution
    """
    # Check if this is an SSH-related script
    if "ssh" in script_content.lower():
        # Extract SSH connection details using regex
        ssh_pattern = r'ssh\s+(?:-\w+\s+)*(?:(\w+)@)?([^\s]+)'
        matches = re.findall(ssh_pattern, script_content)
        
        if matches:
            user, host = matches[-1]  # Take the last SSH command in case of multiple
            
            # Check if this is a ControlMaster setup command
            if "ControlMaster" in script_content and execution_result.get("success", False):
                control_path = re.findall(r'ControlPath=([^\s]+)', script_content)
                if control_path:
                    st.session_state.ssh_session.update({
                        "active": True,
                        "host": host,
                        "user": user,
                        "control_path": control_path[0]
                    })
            
            # Store the last command output
            if execution_result.get("output"):
                st.session_state.ssh_session["last_command_output"] = execution_result["output"]
            
            # Check for session termination
            if "ControlMaster=no" in script_content or "exit" in script_content.lower():
                st.session_state.ssh_session.update({
                    "active": False,
                    "host": None,
                    "user": None,
                    "control_path": None
                })


def main():
    """Main function that handles the Streamlit chat interface and interaction flow"""
    # Initialize session state and render the header
    init_session_state()
    render_header()

    # Check if base components are loaded, if not load them
    if "base_components_loaded" not in st.session_state:
        with st.spinner("Loading base components..."):
            if initialize_base_components():
                st.session_state.base_components_loaded = True
                st.success("Base components loaded successfully!")
            else:
                st.error("Failed to load base components")
                return
    
    # Render controls for generation parameters and handle pipeline selection
    render_generation_controls() 
    handle_pipeline_switch()

    # Create context selection dropdown and handle context changes
    context = st.selectbox("Select Context", ["None", "File"], index=0)
    handle_context_switch(context)

    # Display SSH session status if active
    if st.session_state.ssh_session["active"]:
        st.sidebar.success(f"🔗 Active SSH Session: {st.session_state.ssh_session['user']}@{st.session_state.ssh_session['host']}")
    
    # Display chat history from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=os.path.join("shared", "assets", "botpic.jpg") if message["role"] == "assistant" else None):
            display_chat_message(message)

    # Handle new user input
    if prompt := st.chat_input("Ask me anything!"):
        # If SSH session is active, prepend session info to prompt
        if st.session_state.ssh_session["active"]:
            prompt = f"Using the existing SSH session ({st.session_state.ssh_session['user']}@{st.session_state.ssh_session['host']}), {prompt}"
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "format": "text"
        })

        # Generate and display assistant response
        with st.spinner("Generating response..."):
            reply = generate_response(prompt)
            
            try:
                # Parse the response
                if isinstance(reply, str):
                    response_data = json.loads(reply)
                else:
                    response_data = reply
                
                llm_output = response_data.get("llm_output", "")
                execution_status = response_data.get("execution_status", [])
                
                # Extract only the code block from the LLM output
                code_blocks = re.findall(r'```(?:bash|sh)\n(.*?)\n```', llm_output, re.DOTALL)
                if code_blocks:
                    # Display only the code block
                    with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                        st.code(code_blocks[0], language="bash")
                    
                    # Store the script with proper formatting
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": code_blocks[0],
                        "format": "code"
                    })
                
                    # Display Kali output if script was executed
                    if execution_status and execution_status[0].get("executed_on_kali"):
                        execution_result = execution_status[0].get("execution_result", {})
                        
                        # Update SSH session state based on script execution
                        update_ssh_session_state(code_blocks[0], execution_result)
                        
                        # Create a formatted output string
                        output_parts = []
                        if execution_result.get("output"):
                            output_parts.append("=== Command Output ===\n" + execution_result["output"])
                        if execution_result.get("error"):
                            output_parts.append("=== Error Output ===\n" + execution_result["error"])
                        
                        if output_parts:
                            kali_output = "\n".join(output_parts)
                            st.markdown("**Kali Linux Output:**")
                            st.code(kali_output, language="bash")
                            
                            # Store the Kali output with proper formatting
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": kali_output,
                                "format": "code"
                            })
                else:
                    # No code blocks found in the response
                    with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                        st.error("No bash script found in the response")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Error: No bash script found in the response",
                        "format": "text"
                    })
                        
            except json.JSONDecodeError as e:
                error_msg = f"Error: Invalid response format - {str(e)}"
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Raw response: {reply}")
                with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                    st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "format": "text"
                })
            except Exception as e:
                error_msg = f"Error processing response: {str(e)}"
                logger.error(f"Unexpected error: {e}")
                logger.error(f"Raw response: {reply}")
                with st.chat_message("assistant", avatar=os.path.join("shared", "assets", "botpic.jpg")):
                    st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "format": "text"
                })


if __name__ == "__main__":
    main()