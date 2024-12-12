import streamlit as st
import requests
import time
# Backend API URLs
BASE_URL = "http://127.0.0.1:7575"
LOAD_MODEL_URL = f"{BASE_URL}/load_model"
LOAD_PIPELINE_URL = f"{BASE_URL}/load_pipeline"
LOAD_DOCS_URL = f"{BASE_URL}/load_docs"
LOAD_ENSEMBLE_RETRIEVER_URL = f"{BASE_URL}/load_ensemble_retriever_from_docs"
LOAD_CHAIN_URL = f"{BASE_URL}/load_chain"
GENERATE_URL = f"{BASE_URL}/generate"

# Initialize Streamlit app
st.logo("Logo.png", size='large') 
st.image("Logo.png", width=100)
st.title("Cyber Rabbit")

# Context selector
context = st.selectbox("Select Context", ["File", "Web"], index=0)

# Reload only when switching between contexts
if "current_context" not in st.session_state or st.session_state.current_context != context:
    with st.spinner(f"Switching context to '{context}' and reloading..."):
        try:
            # Reload model and pipeline for any context change
            requests.post(LOAD_MODEL_URL)
            requests.post(LOAD_PIPELINE_URL, json={"task_type": "text-generation"})

            # For "File" context, load documents and ensemble retriever
            if context == "File":
                requests.post(LOAD_DOCS_URL)
                requests.post(LOAD_ENSEMBLE_RETRIEVER_URL)

            st.session_state.current_context = context
            st.success(f"Switched to '{context}' context successfully!")
        except Exception as e:
            st.error(f"Failed to switch context: {e}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and generate response
if prompt := st.chat_input("Ask me anything!"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate a response without reloading the model or retriever
    with st.spinner("Generating response..."):
        try:
            response = requests.post(GENERATE_URL, json={"question": prompt}).json()
            reply = response.get("response", "Sorry, I couldn't generate a response.")
        

    # Display assistant response
            with st.chat_message("assistant", avatar="botpic.jpg"):
                chat_placeholder = st.empty()
                for i in range(len(reply)):
                    chat_placeholder.markdown(reply[:i+1], unsafe_allow_html=True)
                    time.sleep(0.05)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            reply = f"Error generating response: {e}"
            st.error(reply)

# import streamlit as st
# from unittest.mock import patch
# import time

# # Mock API function
# def mock_post(url, json=None):
#     """Mock API responses for testing the Streamlit app independently."""
#     if "/load_model" in url:
#         return MockResponse({"message": "Model loaded successfully!"}, 200)
#     elif "/load_pipeline" in url:
#         return MockResponse({"message": "Pipeline loaded successfully!"}, 200)
#     elif "/load_docs" in url:
#         return MockResponse({"message": "Documents loaded successfully!"}, 200)
#     elif "/load_ensemble_retriever_from_docs" in url:
#         return MockResponse({"message": "Ensemble retriever loaded successfully!"}, 200)
#     elif "/load_chain" in url:
#         return MockResponse({"message": "Chain loaded successfully!"}, 200)
#     elif "/generate" in url:
#         question = json.get("question", "No question provided")
#         return MockResponse({"response": f"Mocked response for: {question}"}, 200)
#     return MockResponse({"error": "Invalid URL"}, 404)

# # MockResponse class to simulate requests.Response
# class MockResponse:
#     """Mock class to simulate the behavior of `requests.Response`."""
#     def __init__(self, json_data, status_code):
#         self.json_data = json_data
#         self.status_code = status_code

#     def json(self):
#         return self.json_data

# # Streamlit app with mocked backend
# @patch("requests.post", side_effect=mock_post)
# def test_streamlit_app(mock_post_func):
#     st.logo("Logo.png", size='large')
#     st.image("Logo.png", width=100)
#     st.title("Cyber Rabbit (Test)")

#     # Context selector
#     context = st.selectbox("Select Context", ["File", "Web"], index=0)

#     # Reload only when switching between contexts
#     if "current_context" not in st.session_state or st.session_state.current_context != context:
#         with st.spinner(f"Switching context to '{context}' and reloading..."):
#             try:
#                 # Reload model and pipeline for any context change
#                 response_model = mock_post("/load_model").json()
#                 response_pipeline = mock_post("/load_pipeline", json={"task_type": "text-generation"}).json()

#                 # For "File" context, load documents and ensemble retriever
#                 if context == "File":
#                     response_docs = mock_post("/load_docs").json()
#                     response_retriever = mock_post("/load_ensemble_retriever_from_docs").json()

#                 st.session_state.current_context = context
#                 st.success(f"Switched to '{context}' context successfully!")
#             except Exception as e:
#                 st.error(f"Failed to switch context: {e}")

#     # Chat interface
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Accept user input and generate response
#     if prompt := st.chat_input("Ask me anything!"):
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         with st.spinner("Generating response..."):
#             try:
#                 response_generate = mock_post("/generate", json={"question": prompt}).json()
#                 reply = response_generate.get("response", "Sorry, I couldn't generate a response.")
#             except Exception as e:
#                 reply = f"Error generating response: {e}"

#         # Display assistant response
#         with st.chat_message("assistant", avatar="botpic.jpg"):
#             chat_placeholder = st.empty()
#             for i in range(len(reply)):
#                 chat_placeholder.markdown(reply[:i+1], unsafe_allow_html=True)
#                 time.sleep(0.05)  # Adjust the sleep time to control typing speed

#             st.session_state.messages.append({"role": "assistant", "content": reply})

# # Run the test
# if __name__ == "__main__":
#     test_streamlit_app()
