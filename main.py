import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import AIMessageChunk
from langchain_ollama.chat_models import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search import DDGS
import tempfile
import hashlib
import yaml
from pathlib import Path
import json
import datetime
# import requests  # MCP related import

print("Starting the application...")

# Load environment variables
load_dotenv()

# Initialize components
search = DuckDuckGoSearchRun()
embeddings = OllamaEmbeddings(model='nomic-embed-text')
llm = Ollama(model="deepseek-r1:8b", temperature=0.7)




# MCP Server configuration
# MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
# MCP_API_KEY = os.getenv("MCP_API_KEY", "your-api-key-here")  # Default key for local development

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Authentication functions
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.session_state["credentials"]:
            if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == st.session_state["credentials"][st.session_state["username"]]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store password
                del st.session_state["username"]  # Don't store username
            else:
                st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        return True

def load_credentials():
    """Load credentials from config file or create default if not exists."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        default_credentials = {
            "admin": hashlib.sha256("admin".encode()).hexdigest()
        }
        with open(config_path, "w") as f:
            yaml.dump(default_credentials, f)
        return default_credentials
    else:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

# Load credentials
st.session_state["credentials"] = load_credentials()

def process_documents(uploaded_files):
    """Process uploaded documents and create a vector store."""
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
            
            if file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif file.name.endswith('.docx'):
                loader = Docx2txtLoader(tmp_file_path)
            else:
                st.error(f"Unsupported file type: {file.name}")
                continue
                
            docs = loader.load()
            all_docs.extend(docs)
            
        os.unlink(tmp_file_path)
    
    if not all_docs:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)
    return FAISS.from_documents(splits, embeddings)

def process_web_url(url):
    """Process web URL and create a vector store."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=50
        )
        splits = text_splitter.split_documents(docs)
        return FAISS.from_documents(splits, embeddings)
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        return None

def refine_query_with_pdf(query, vector_store):
    """Refine the query based on PDF content and create a more focused search prompt."""
    if not vector_store:
        return query
    
    # First, get relevant information from PDFs
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    
    # Get context from PDFs
    pdf_context = qa_chain.run(query)
    
    # Create a refined prompt that combines the original query with PDF context
    refined_prompt = f"""
    Based on the following context from PDF documents:
    {pdf_context}
    
    Original query: {query}
    
    Please provide a refined search query that combines the PDF context with the original query to get more specific and relevant web search results. The search query should be concise and to the point.
    """
    
    # Get the refined query from the LLM
    refined_query = llm.invoke(refined_prompt)
    
    
    
    return refined_query, pdf_context

# def call_mcp_server(query, context=None):
#     """Call the MCP server with the query and optional context."""
#     try:
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {MCP_API_KEY}"
#         }
#         
#         payload = {
#             "query": query,
#             "context": context if context else ""
#         }
#         
#         response = requests.post(
#             f"{MCP_SERVER_URL}/query",
#             headers=headers,
#             json=payload
#         )
#         
#         if response.status_code == 200:
#             return response.json()
#         else:
#             st.error(f"MCP Server Error: {response.status_code} - {response.text}")
#             return None
#     except Exception as e:
#         st.error(f"Error calling MCP server: {str(e)}")
#         return None

def get_agent_response(query, vector_store):
    """Get response from the agent using both PDF knowledge and web search."""
    # First, refine the query using PDF content
    refined_query, pdf_context = refine_query_with_pdf(query, vector_store)
    
    try:
        with DDGS() as ddgs:
            web_results = ddgs.text(refined_query)
            for result in web_results:
                print(result)
            web_results = json.dumps(web_results)
    except Exception as e:
        print(f"Search failed: {e}")
        web_results = refined_query
    


    
    # Add to current conversation
    st.session_state.current_conversation.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    


    
    # Combine responses
    combined_prompt = f"""
    You are a helpful assistant that can answer questions and help with tasks.
    Based on the following context make a summary of the information and answer the question.
    
    Original query:
    {query}
    
    PDF Context:
    {pdf_context}
    
    Refined Search Query:
    {refined_query}
    
    Web Search Results:
    {web_results}
    
    Past Conversation:
    {st.session_state.current_conversation}
    
    """
    
    # Create a placeholder for the streaming response
    response_placeholder = st.empty()
    full_response = ""
    
    # Stream the response
    for chunk in llm.stream(combined_prompt):
        if chunk:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
    
    # Update the placeholder with the final response
    response_placeholder.markdown(full_response, unsafe_allow_html=True)
    
    # Add assistant response to conversation
    st.session_state.current_conversation.append({
        "role": "assistant",
        "content": full_response,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return full_response

def reset_conversation():
    """Reset the current conversation and save it to history."""
    if st.session_state.current_conversation:
        st.session_state.conversation_history.append({
            "id": len(st.session_state.conversation_history) + 1,
            "conversation": st.session_state.current_conversation.copy(),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    st.session_state.current_conversation = []
    st.session_state.vector_store = None

# Main app
def main():
    st.title("Document Processing & Web Search Demo")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Chat", "History"])
    
    with tab1:
        # Reset button
        if st.button("Reset Conversation"):
            reset_conversation()
            st.rerun()
        
        # Input section
        st.header("Input")
        query = st.text_input("Enter your query:")
        uploaded_files = st.file_uploader("Upload documents (PDF or DOCX)", accept_multiple_files=True)
        
        # Process documents
        if uploaded_files:
            with st.spinner("Processing documents..."):
                st.session_state.vector_store = process_documents(uploaded_files)
                if st.session_state.vector_store:
                    st.success("Documents processed successfully!")
        
        # Display current conversation
        st.header("Current Conversation")
        for message in st.session_state.current_conversation:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                st.caption(message["timestamp"])
        
        # Query processing
        if st.button("Submit Query"):
            if query:
                with st.spinner("Processing query..."):
                    response = get_agent_response(query, st.session_state.vector_store)
            else:
                st.warning("Please enter a query.")
    
    with tab2:
        st.header("Conversation History")
        for conversation in reversed(st.session_state.conversation_history):
            with st.expander(f"Conversation {conversation['id']} - {conversation['timestamp']}"):
                for message in conversation["conversation"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)
                        st.caption(message["timestamp"])

# Run the app with authentication
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

main() 