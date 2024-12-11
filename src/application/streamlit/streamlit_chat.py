import streamlit as st
import requests  

# Backend API URLs
BASE_URL = "http://127.0.0.1:7575"  
LOAD_MODEL_URL = f"{BASE_URL}/load_model"
LOAD_PIPELINE_URL = f"{BASE_URL}/load_pipeline"
LOAD_CHAIN_URL = f"{BASE_URL}/load_chain"
GENERATE_URL = f"{BASE_URL}/generate"

# Initialize Streamlit app
st.logo("Logo.png", size='large')
st.image("Logo.png", width=100)
st.title("Cyber Rabbit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load model, pipeline, and chain in the background
if "initialized" not in st.session_state:
    with st.spinner("Initializing..."):
        try:
            # Load model
            requests.post(LOAD_MODEL_URL)
            # Load pipeline
            requests.post(LOAD_PIPELINE_URL, json={"task_type": "text-generation"})
            # Load chain
            requests.post(LOAD_CHAIN_URL, json={"use_hyde": False})
            st.session_state.initialized = True
            st.success("initialized!")
        except Exception as e:
            st.error(f"Initialization failed: {e}")

# Accept user input
if prompt := st.chat_input("What's cookin' good lookin'?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response via API call
    with st.spinner("Generating response..."):
        try:
            response = requests.post(GENERATE_URL, json={"question": prompt}).json()
            reply = response.get("response", "Sorry, I couldn't generate a response.")
        except Exception as e:
            reply = f"Error generating response: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="botpic.jpg"):
        st.markdown(reply)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": reply})
# import streamlit as st
# import requests
# from unittest.mock import patch  # For mocking the requests library

# # Backend API URLs (Fake for testing)
# BASE_URL = "http://fake-api-server"
# GENERATE_URL = f"{BASE_URL}/generate"

# # Function to mock API response
# def fake_generate_api_call(url, json):
#     """
#     Simulates a fake API response for testing.
    
#     Args:
#         url (str): The URL being called (ignored in this mock).
#         json (dict): The payload sent to the API, expected to contain the question.

#     Returns:
#         MockResponse: A mock response object mimicking `requests.Response`.
#     """
#     class MockResponse:
#         def json(self):
#             question = json.get("question", "")
#             return {"response": f"This is a fake response to: {question}"}

#     return MockResponse()

# # Initialize Streamlit app
# st.title("Testing Fake API Calls")

# # Accept user input
# if prompt := st.text_input("Enter your prompt:"):
#     with st.spinner("Generating response..."):
#         try:
#             # Mock the API call for testing
#             with patch("requests.post", side_effect=fake_generate_api_call):
#                 response = requests.post(GENERATE_URL, json={"question": prompt}).json()
#                 reply = response.get("response", "Sorry, I couldn't generate a response.")
#         except Exception as e:
#             reply = f"Error generating response: {e}"

#     # Display the mock response
#     st.markdown(f"**Assistant:** {reply}")
