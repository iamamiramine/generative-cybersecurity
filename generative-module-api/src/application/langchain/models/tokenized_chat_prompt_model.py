import json

from langchain_core.prompts import PromptTemplate

class BaseTokenizedChatPromptTemplate:
    """
    Base class for tokenized chat prompt templates.
    Provides common functionality for loading prompt configurations and tokenization.
    """
    def __init__(self, tokenizer, input_variables, prompt_config_path="shared/config/prompts.json"):
        """
        Initialize the base prompt template.

        Args:
            tokenizer: The tokenizer to use for processing prompts
            input_variables: List of variables that will be used in the prompt template
            prompt_config_path: Path to JSON config file containing prompt templates
        """
        self.tokenizer = tokenizer
        self.input_variables = input_variables
        
        # Load prompt configurations from JSON file
        with open(prompt_config_path, 'r') as f:
            self.prompt_config = json.load(f)

    def get_prompt(self):
        """
        Abstract method to get the formatted prompt template.
        Must be implemented by subclasses.
        
        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement get_prompt method")

class BasicTokenizedChatPromptTemplate(BaseTokenizedChatPromptTemplate):
    """
    A basic tokenized chat prompt template that handles simple question-answer interactions.
    Inherits from BaseTokenizedChatPromptTemplate to provide basic prompt functionality.
    """
    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        """
        Initialize the basic prompt template.

        Args:
            tokenizer: The tokenizer to use for processing prompts
            prompt_config_path: Path to JSON config file containing prompt templates (default: "shared/config/prompts.json")
        """
        super().__init__(tokenizer, input_variables=["question"], prompt_config_path=prompt_config_path)

    def get_prompt(self):
        """
        Constructs and returns a formatted prompt template for basic question-answer interactions.
        
        The prompt combines:
        1. A system base message from config
        2. A question placeholder
        3. An answer prefix for bash scripts
        
        Returns:
            PromptTemplate: A formatted template ready for question input
        """
        # Get the base system message from config
        system = self.prompt_config["basic"]["system_base"]
        # Add question placeholder
        question = """<|im_start|>Question: {question}<|im_end|>\n"""
        # Add answer prefix
        answer = """<|im_start|>Bash Script: <|im_end|>\n"""
        # Combine all parts
        system = system + question + answer
        
        return PromptTemplate.from_template(system)

class RAGTokenizedChatPromptTemplate(BaseTokenizedChatPromptTemplate):
    """
    A Retrieval-Augmented Generation (RAG) prompt template that incorporates context and questions.
    Inherits from BaseTokenizedChatPromptTemplate to provide RAG-specific prompt functionality.
    """
    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        """
        Initialize the RAG prompt template.

        Args:
            tokenizer: The tokenizer to use for processing prompts
            prompt_config_path: Path to JSON config file containing prompt templates (default: "shared/config/prompts.json")
        """
        super().__init__(tokenizer, input_variables=["context", "question"], prompt_config_path=prompt_config_path)

    def get_prompt(self):
        """
        Constructs and returns a formatted prompt template for RAG-based interactions.
        
        The prompt combines:
        1. A system base message from config defining the bash expert role
        2. Retrieved context from the RAG system
        3. The user's question
        4. An answer prefix for bash scripts
        
        Each section is wrapped with <|im_start|> and <|im_end|> tokens for proper formatting.

            RAG Prompt Template
                <|im_start|>You are an expert in bash scripting. 
                Given a question about bash commands, please provide an answer with bash scripts only, 
                and make sure to format with codeblocks using ```bash and ```<|im_end|>.

                <|im_start|>Context: {context}<|im_end|>

                <|im_start|>Question: {question}<|im_end|>

                <|im_start|>Bash Script: 
        
        Returns:
            PromptTemplate: A formatted template ready for context and question inputs
        """
        # Get the base system message from config that defines the AI's role
        system_base = self.prompt_config["rag"]["system_base"]
        
        # Format the context section with special tokens
        context = """<|im_start|>Context: {context}<|im_end|>\n"""
        
        # Format the question section with special tokens
        question = """<|im_start|>Question: {question}<|im_end|>\n"""
        
        # Add the answer prefix with special tokens
        answer = """<|im_start|>Bash Script: """
        
        # Combine all parts in the correct order
        system = system_base + context + question + answer

        return PromptTemplate.from_template(system)

class HYDEGenerationPromptTemplate(BaseTokenizedChatPromptTemplate):
    """
    A prompt template class for generating hypothetical documents using HYDE (Hypothetical Document Embeddings).
    
    This template is used in the first step of HYDE where we generate a hypothetical document
    that could potentially answer the user's question. This document will later be used for 
    retrieval augmentation.
    """
    
    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        """
        Initialize the HYDE generation prompt template.
        
        Args:
            tokenizer: The tokenizer to use for processing prompts
            prompt_config_path (str): Path to JSON config file containing prompt templates
                                    (default: "shared/config/prompts.json")
        """
        super().__init__(tokenizer, input_variables=["question"], prompt_config_path=prompt_config_path)

    def get_prompt(self):
        """
        Constructs and returns a formatted prompt template for HYDE document generation.
        
        The prompt combines:
        1. A system base message from config
        2. A HYDE-specific template base
        3. The user's question
        4. A prefix for the hypothetical document
        
        Returns:
            PromptTemplate: A formatted template ready for the question input
        """
        # Get the base system message from config
        system = self.prompt_config["hyde_generation"]["system_base"]
        
        # Get the HYDE-specific template instructions
        hyde_base = self.prompt_config["hyde_generation"]["hyde_template_base"]
        
        # Format the question section
        question = """Question: {question}\n"""
        
        # Add the hypothetical document prefix
        hyptohetical_document = """Hypothetical Document: """
        
        # Combine all parts in the correct order
        hyde_template = system + hyde_base + question + hyptohetical_document
        
        return PromptTemplate.from_template(hyde_template)

class HYDEFinalPromptTemplate(BaseTokenizedChatPromptTemplate):
    """
    A prompt template class for generating the final response using HYDE (Hypothetical Document Embeddings).
    
    This template is used in the final step of HYDE where we combine the hypothetical document,
    retrieved context, and original question to generate a bash script response.
    """
    
    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        """
        Initialize the HYDE final prompt template.
        
        Args:
            tokenizer: The tokenizer to use for processing prompts
            prompt_config_path (str): Path to JSON config file containing prompt templates
                                    (default: "shared/config/prompts.json")
        """
        super().__init__(tokenizer, input_variables=["question", "hypothetical_document", "context"], prompt_config_path=prompt_config_path)

    def get_prompt(self):
        """
        Constructs and returns a formatted prompt template for the final HYDE response.
        
        The prompt combines:
        1. A system base message from config
        2. The generated hypothetical document
        3. Retrieved context from the knowledge base
        4. The original user question
        5. A prefix for the bash script response
        
        Returns:
            PromptTemplate: A formatted template ready for all required inputs
        """
        # Get the base system message from config
        system_base = self.prompt_config["hyde_final"]["system_base"]
        
        # Format the hypothetical document section
        hyptohetical_document = """Hypothetical Document: {hypothetical_document}\n"""
        
        # Format the retrieved context section
        context = """Retrieved Context: {context}\n"""
        
        # Format the question section
        question = """Question: {question}\n"""
        
        # Add the bash script prefix
        answer = """Bash Script: """
        
        # Combine all parts in the correct order
        hyde_final_template = system_base + hyptohetical_document + context + question + answer
        
        return PromptTemplate.from_template(hyde_final_template)
