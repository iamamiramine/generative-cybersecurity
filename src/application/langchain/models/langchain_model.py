from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import ChatMessage
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)

from application.local_llm.services.local_llm_service import generate, chat

class LangChainLocalLLM(LLM):
    """LangChain wrapper for text completion with LocalLLMAgent"""

    llm_url: str
    
    def __init__(self, llm_url: str):
        super().__init__()
        self.llm_url = llm_url

    @property
    def _llm_type(self) -> str:
        return "local_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the local LLM to generate text completion.

        Args:
            prompt (str): The text prompt to generate completion for
            stop (Optional[List[str]], optional): List of strings to stop generation at. Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun], optional): Callback manager for LLM runs. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying generate() function:
                temperature (float, optional): Controls randomness in generation. Higher values (e.g. 0.8) make output more random,
                    lower values (e.g. 0.2) make it more deterministic. Defaults to 0.7.
                max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.
                top_p (float, optional): Nucleus sampling parameter - only consider tokens whose cumulative probability exceeds this value.
                    Defaults to 0.9.

        Returns:
            str: The generated text completion

        Example:
            >>> llm = LangChainLocalLLM(llm_url="http://localhost:5000")
            >>> llm("Write a haiku about cybersecurity")
            'Firewalls standing tall\nProtecting precious data\nSafe from prying eyes'

            >>> # With custom parameters
            >>> llm("Write a creative story", temperature=0.9, max_new_tokens=1024)
            'Once upon a time...'

            >>> # With stop sequences
            >>> llm("List 3 security tips:", stop=["\n"])
            '1. Use strong passwords'
        """
        response = generate(api_url=self.llm_url, prompt=prompt, stop=stop, **kwargs)
        return response["text"]

class LangChainLocalChatModel(BaseChatModel):
    """LangChain wrapper for chat completion with LocalLLMAgent"""

    @property
    def _llm_type(self) -> str:
        return "local_chat_model"

    def _convert_messages_to_langchain(self, message: dict) -> BaseMessage:
        """Convert API messages to LangChain format"""
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            return SystemMessage(content=content)
        elif role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        else:
            raise ValueError(f"Unknown role: {role}")

    def _convert_messages_to_api(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to API format"""
        api_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                api_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                api_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                api_messages.append({"role": "assistant", "content": message.content})
            else:
                raise ValueError(f"Unknown message type: {type(message)}")
        return api_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        api_messages = self._convert_messages_to_api(messages)
        response = chat(api_url=self.llm_url, messages=api_messages, stop=stop, **kwargs)
        return self._convert_messages_to_langchain(response["message"])