import json
from typing import Any
from pydantic import Field

from langchain_core.prompts import StringPromptTemplate


class BaseTokenizedChatPromptTemplate(StringPromptTemplate):
    tokenizer: Any = Field(default=None) 
    prompt_config: dict = Field(default=None)

    def __init__(self, tokenizer, input_variables, prompt_config_path="config/prompts.json"):
        super().__init__(input_variables=input_variables)
        self.tokenizer = tokenizer

        with open(prompt_config_path, 'r') as f:
            self.prompt_config = json.load(f)

    def format(self, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement format method")
    

class BasicTokenizedChatPromptTemplate(BaseTokenizedChatPromptTemplate):
    def __init__(self, tokenizer, prompt_config_path="config/prompts.json"):
        super().__init__(tokenizer, input_variables=["question"], prompt_config_path=prompt_config_path)

    def format(self, **kwargs) -> str:
        system = self.prompt_config["basic"]["system_base"]
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": kwargs["question"]},
            {"role": "assistant", "content": "Bash Script:"}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    

class RAGTokenizedChatPromptTemplate(BaseTokenizedChatPromptTemplate):
    def __init__(self, tokenizer, prompt_config_path="config/prompts.json"):
        super().__init__(tokenizer, input_variables=["context", "question"], prompt_config_path=prompt_config_path)

    def format(self, **kwargs) -> str:  # Add kwargs back
        system_base = self.prompt_config["rag"]["system_base"]
        retrieved_section = f"\nContext: {kwargs['context']}"

        system = system_base + retrieved_section
        chat = [
            {"role": "system", "content": system.format(**kwargs)},
            {"role": "user", "content": kwargs["question"]},
            {"role": "assistant", "content": "Bash Script:"}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


class HYDEGenerationPromptTemplate(BaseTokenizedChatPromptTemplate):
    def __init__(self, tokenizer, prompt_config_path="config/prompts.json"):
        super().__init__(tokenizer, input_variables=["question"], prompt_config_path=prompt_config_path)

    def format(self, **kwargs) -> str:
        system = self.prompt_config["hyde_generation"]["system_base"]
        hyde_base = self.prompt_config["hyde_generation"]["hyde_template_base"]
        question = f"\nQuestion: {kwargs['question']}"
        hypothetical_document = f"\nHypothetical Document:"
        hyde_template = hyde_base + question + hypothetical_document
        
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": hyde_template.format(**kwargs)},
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


class HYDEFinalPromptTemplate(BaseTokenizedChatPromptTemplate):
    def __init__(self, tokenizer, prompt_config_path="config/prompts.json"):
        super().__init__(tokenizer, input_variables=["question", "hypothetical_document", "context"], prompt_config_path=prompt_config_path)

    def format(self, **kwargs) -> str:
        system_base = self.prompt_config["hyde_final"]["system_base"]

        hypothetical_section = f"\n\nHypothetical Document: {kwargs['hypothetical_document']}"
        retrieved_section = f"\nRetrieved Context: {kwargs['context']}"

        system = system_base + hypothetical_section + retrieved_section

        chat = [
            {"role": "system", "content": system.format(**kwargs)},
            {"role": "user", "content": kwargs["question"]},
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
