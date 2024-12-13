# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class LangchainState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.hf = None
        self.ensemble_retriever = None
        self.retrieval_chain = None
        self.chain = None
        self.agent = None
        # self.messages = StreamlitChatMessageHistory(key="langchain_messages")
        self.messages = ChatMessageHistory()
        self.call_counter = 0
        self.docs = None
        self.task = None

    def is_model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def is_pipeline_loaded(self) -> bool:
        return self.hf is not None

    def is_docs_loaded(self) -> bool:
        return self.docs is not None

    def is_retriever_loaded(self) -> bool:
        return self.ensemble_retriever is not None

    def is_chain_loaded(self) -> bool:
        return self.chain is not None