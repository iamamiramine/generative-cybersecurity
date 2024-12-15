from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


def create_memory_chain(llm, base_chain, chat_memory):
    """
    Creates a memory-enabled chain that can contextualize questions based on chat history.
    
    Args:
        llm: The language model to use for question contextualization
        base_chain: Optional base chain to run after question contextualization
        chat_memory: Chat history memory instance to store conversation
        
    Returns:
        RunnableWithMessageHistory: A chain that maintains conversation history and contextualizes questions
    """
    # System prompt that instructs the model how to contextualize questions
    contextualize_q_system_prompt = """Given the chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. Remember, you are trying to understand the user's question to retrieve relevant context and provide a bash script answer."""

    # Create prompt template combining system prompt, chat history, and user question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # If no base chain provided, add bash expert instructions and create simple chain
    if base_chain == None:
        contextualize_q_prompt += "You are an expert in bash scripting. Given a question about bash commands, please provide an answer with bash scripts only, and make sure to format with codeblocks using ```bash and ```"
        runnable = contextualize_q_prompt | llm
    # Otherwise chain the base_chain after question contextualization
    else:
        runnable = contextualize_q_prompt | llm | base_chain

    # Function to retrieve chat history for a given session
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    # Create memory-enabled chain that tracks conversation history
    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return with_message_history
