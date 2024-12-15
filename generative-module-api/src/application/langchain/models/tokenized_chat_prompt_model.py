import json
from typing import Any
from pydantic import Field

from langchain_core.prompts import StringPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

class BaseTokenizedChatPromptTemplate:
    def __init__(self, tokenizer, input_variables, prompt_config_path="shared/config/prompts.json"):
        self.tokenizer = tokenizer
        self.input_variables = input_variables
        
        with open(prompt_config_path, 'r') as f:
            self.prompt_config = json.load(f)

    def get_prompt(self) -> ChatPromptTemplate:
        raise NotImplementedError("Subclasses must implement get_prompt method")

class BasicTokenizedChatPromptTemplate(BaseTokenizedChatPromptTemplate):
    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        super().__init__(tokenizer, input_variables=["question"], prompt_config_path=prompt_config_path)

    def get_prompt(self) -> ChatPromptTemplate:
        system = self.prompt_config["basic"]["system_base"]
        
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
            ("assistant", "Bash Script:")
        ])

class RAGTokenizedChatPromptTemplate(BaseTokenizedChatPromptTemplate):
    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        super().__init__(tokenizer, input_variables=["context", "question"], prompt_config_path=prompt_config_path)

    def get_prompt(self) -> ChatPromptTemplate:
        system_base = """<|im_start|>You are an expert in bash scripting. Given a question about bash commands, please provide an answer with bash scripts only, and make sure to format with codeblocks using ```bash and ```<|im_end|>."""
        
        context = """<|im_start|>Context: {context}<|im_end|>\n"""
        
        # question = """<|im_start|>Question: {question}<|im_end|>"""

        system = system_base + context

        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
            ("assistant", "Bash Script:")
        ])

class HYDEGenerationPromptTemplate(BaseTokenizedChatPromptTemplate):
    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        super().__init__(tokenizer, input_variables=["question"], prompt_config_path=prompt_config_path)

    def get_prompt(self) -> ChatPromptTemplate:
        system = self.prompt_config["hyde_generation"]["system_base"]
        hyde_base = self.prompt_config["hyde_generation"]["hyde_template_base"]
        hyde_template = f"{hyde_base}\nQuestion: {{question}}\nHypothetical Document:"
        
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", hyde_template)
        ])

class HYDEFinalPromptTemplate(BaseTokenizedChatPromptTemplate):
    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        super().__init__(tokenizer, input_variables=["question", "hypothetical_document", "context"], prompt_config_path=prompt_config_path)

    def get_prompt(self) -> ChatPromptTemplate:
        system_base = self.prompt_config["hyde_final"]["system_base"]
        system = f"{system_base}\n\nHypothetical Document: {{hypothetical_document}}\nRetrieved Context: {{context}}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}")
        ])