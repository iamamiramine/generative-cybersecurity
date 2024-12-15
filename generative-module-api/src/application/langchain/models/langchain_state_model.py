from langchain_community.chat_message_histories import ChatMessageHistory

class LangchainState:
    """
    Class to manage the state of Langchain components and resources.
    Tracks loading status of models, retrievers, chains and other resources.
    """
    def __init__(self):
        # Core model components
        self.model = None  # The main language model
        self.tokenizer = None  # Tokenizer for the model
        self.hf = None  # HuggingFace pipeline
        
        # Retrieval components
        self.ensemble_retriever = None  # Combined retriever
        self.retrieval_chain = None  # Chain for retrieval operations
        self.chain = None  # Main processing chain
        self.agent = None  # Agent for task execution
        
        # Chat history and state tracking
        self.messages = ChatMessageHistory()  # Store conversation history
        self.call_counter = 0  # Track number of API calls
        self.docs = None  # Loaded documents
        self.task = None  # Current task being processed

    def is_model_loaded(self) -> bool:
        """Check if both model and tokenizer are loaded"""
        return self.model is not None and self.tokenizer is not None

    def is_pipeline_loaded(self) -> bool:
        """Check if HuggingFace pipeline is initialized"""
        return self.hf is not None

    def is_docs_loaded(self) -> bool:
        """Check if documents are loaded into memory"""
        return self.docs is not None

    def is_retriever_loaded(self) -> bool:
        """Check if ensemble retriever is initialized"""
        return self.ensemble_retriever is not None

    def is_chain_loaded(self) -> bool:
        """Check if main processing chain is ready"""
        return self.chain is not None