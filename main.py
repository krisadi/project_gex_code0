import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import hashlib
import yaml
from pathlib import Path
import datetime
import tempfile
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import socket
import requests
from urllib3.exceptions import NewConnectionError
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import json
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

# Check for Tavily API key
if not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please add it to your .env file.")

print("Starting the Financial Assistant...")

# Initialize components
embeddings = OllamaEmbeddings(model='nomic-embed-text')
llm = Ollama(model="deepseek-r1:8b", temperature=0.7)
llm_tool = Ollama(model="llama3.1:8b", temperature=0.7)

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    transaction_data: pd.DataFrame
    policy_store: FAISS
    user_profile: dict
    user_transactions: pd.DataFrame

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = []
if "transaction_data" not in st.session_state:
    st.session_state.transaction_data = None
if "policy_store" not in st.session_state:
    st.session_state.policy_store = None
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": "",
        "age": 0,
        "income": 0,
        "financial_goals": [],
        "risk_tolerance": "medium",
        "investment_preferences": [],
        "first_name": "",
        "last_name": "",
        "city": "",
        "country": "United States"
    }
if "user_transactions" not in st.session_state:
    st.session_state.user_transactions = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None

def load_credentials():
    """Load credentials and user profiles from config file or create default if not exists."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        default_config = {
            "users": {
                "admin": {
                    "password": hashlib.sha256("admin".encode()).hexdigest(),
                    "profile": {
                        "name": "Admin User",
                        "first_name": "Admin",
                        "last_name": "User",
                        "city": "Default City",
                        "country": "United States",
                        "age": 30,
                        "income": 50000,
                        "financial_goals": ["Retirement", "Investment"],
                        "risk_tolerance": "medium",
                        "investment_preferences": ["Stocks", "Bonds"]
                    }
                }
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(default_config, f)
        return default_config
    else:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

def save_credentials(config):
    """Save updated credentials and user profiles to config file."""
    config_path = Path("config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def signup():
    """Handle user signup."""
    with st.form("signup_form"):
        st.header("Create New Account")
        username = st.text_input("Choose Username")
        password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        # Profile information
        st.subheader("Profile Information")
        name = st.text_input("Full Name")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        city = st.text_input("City")
        country = st.text_input("Country", value="United States")
        age = st.number_input("Age", min_value=18, max_value=100)
        income = st.number_input("Annual Income", min_value=0)
        financial_goals = st.multiselect(
            "Financial Goals",
            ["Retirement", "Home Purchase", "Education", "Investment", "Debt Reduction", "Emergency Fund"]
        )
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["low", "medium", "high"]
        )
        investment_preferences = st.multiselect(
            "Investment Preferences",
            ["Stocks", "Bonds", "Mutual Funds", "ETFs", "Real Estate", "Cryptocurrency"]
        )
        
        if st.form_submit_button("Sign Up"):
            if password != confirm_password:
                st.error("Passwords do not match!")
                return False
            
            config = load_credentials()
            if username in config["users"]:
                st.error("Username already exists!")
                return False
            
            # Add new user to config
            config["users"][username] = {
                "password": hashlib.sha256(password.encode()).hexdigest(),
                "profile": {
                    "name": name,
                    "first_name": first_name,
                    "last_name": last_name,
                    "city": city,
                    "country": country,
                    "age": age,
                    "income": income,
                    "financial_goals": financial_goals,
                    "risk_tolerance": risk_tolerance,
                    "investment_preferences": investment_preferences
                }
            }
            
            save_credentials(config)
            st.success("Account created successfully! Please log in.")
            return True
    return False

def login():
    """Handle user login."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        config = load_credentials()
        if st.session_state["username"] in config["users"]:
            if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == config["users"][st.session_state["username"]]["password"]:
                st.session_state["password_correct"] = True
                st.session_state["current_user"] = st.session_state["username"]
                # Load user profile
                st.session_state.user_profile = config["users"][st.session_state["username"]]["profile"]
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

def process_policy_documents(uploaded_files):
    """Process uploaded policy documents and create a vector store."""
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
            
            if file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
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

def load_transaction_data():
    """Load and preprocess transaction data from CSV."""
    try:
        df = pd.read_csv('data/CreditCardData.csv')
        # Convert date and time columns to datetime
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        # Sort by transaction date and time
        df = df.sort_values('trans_date_trans_time', ascending=False)
        return df
    except Exception as e:
        st.error(f"Error loading transaction data: {str(e)}")
        return None

def get_user_transactions(df, first_name, last_name, city, country):
    """Extract transactions for a specific user based on first name, last name, city, and country."""
    try:
        # Filter transactions for the specific user
        user_transactions = df[
            (df['first'].str.lower() == first_name.lower()) &
            (df['last'].str.lower() == last_name.lower()) &
            (df['city'].str.lower() == city.lower())
        ]
        
        print(user_transactions)
        
        if user_transactions.empty:
            return None
        
        # Format the transactions for better readability
        formatted_transactions = user_transactions[[
            'trans_date_trans_time',
            'merchant',
            'category',
            'amt',
            'trans_num'
        ]].copy()
        
        # Format the amount as currency
        formatted_transactions['amt'] = formatted_transactions['amt'].apply(lambda x: f"${x:,.2f}")
        
        # Format the date and time
        formatted_transactions['trans_date_trans_time'] = formatted_transactions['trans_date_trans_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return formatted_transactions
    except Exception as e:
        st.error(f"Error extracting user transactions: {str(e)}")
        return None

def get_account_summary(query, df):
    """Generate account summary from transaction data."""
    
    try:
        if df is None:
            return "No transaction data available."
        
        # Calculate summary statistics
        total_transactions = len(df)
        total_spent = df['amt'].str.replace('$', '').str.replace(',', '').astype(float).sum()
        avg_transaction = total_spent / total_transactions if total_transactions > 0 else 0
        
        # Get top categories
        top_categories = df['category'].value_counts().to_dict()
        
        # Get recent transactions
        recent_transactions = df.head(100).to_dict('records')
        
        summary = {
            "total_transactions": total_transactions,
            "total_spent": f"${total_spent:,.2f}",
            "average_transaction": f"${avg_transaction:,.2f}",
            "top_categories": top_categories,
        }
        
        advice_prompt = f"""
            You are a financial advisor. Please answer the user's query the you can :
            
            User Query: {query}
            
            Account Summary:
            {summary}
            
            """        
            
        response_placeholder = st.empty()
        full_response = "Account Summary:\n"
        for chunk in llm.stream(advice_prompt):
            if chunk:
                full_response += chunk
                response_placeholder.markdown(full_response)    
        
        return full_response

    except Exception as e:
        return f"Error generating account summary: {str(e)}"

def save_investment_opportunities(investment_opportunities, user_profile, query, final_advice, conversation_history):
    """Save investment opportunities and related information to a JSON file."""
    try:
        # Create records directory if it doesn't exist
        records_dir = "investment_records"
        if not os.path.exists(records_dir):
            os.makedirs(records_dir)
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{records_dir}/investment_opportunities_{timestamp}.json"
        
        # Prepare data to save
        data_to_save = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_profile": user_profile,
            "query": query,
            "conversation_history": conversation_history,
            "investment_opportunities": investment_opportunities,
            "final_advice": final_advice
        }
        
        # Save to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            
        return True
    except Exception as e:
        st.error(f"Error saving investment opportunities: {str(e)}")
        return False

def get_financial_advice(query, df, user_profile):
    """Generate personalized financial advice based on user profile and transaction history."""
    try:
        # Calculate spending patterns
        if df is not None:
            total_spent = df['amt'].str.replace('$', '').str.replace(',', '').astype(float).sum()
            avg_transaction = total_spent / len(df) if len(df) > 0 else 0
            top_categories = df['category'].value_counts().head(3).to_dict()
        else:
            total_spent = 0
            avg_transaction = 0
            top_categories = {}
        
        # Get current investment opportunities with Tavily Search
        search = TavilySearch(max_results=1, topic="general",)
        
        # Create more comprehensive search prompts with location information
        search_prompts = [
            # Risk tolerance based searches with location
            f"current investment opportunities for {user_profile['risk_tolerance']} risk tolerance investors in {user_profile['city']} {user_profile['country']} 2025",
            f"best {user_profile['risk_tolerance']} risk investment strategies in {user_profile['country']} 2025",
            f"{user_profile['risk_tolerance']} risk portfolio allocation for {user_profile['city']} residents 2025",
        ]
        
        # Perform searches and combine results
        investment_opportunities = "\nCurrent Investment Research:\n"
        seen_results = set()  # To avoid duplicates
        
        for prompt in search_prompts:
            try:
                results = search.run(prompt)
                # Process Tavily search results
                if isinstance(results, list):
                    for result in results:
                        # Create a unique identifier for the result
                        result_id = f"{result.get('title', '')}_{result.get('url', '')}"
                        if result_id not in seen_results:
                            seen_results.add(result_id)
                            investment_opportunities += f"\nSearch: {prompt}\n"
                            investment_opportunities += f"Title: {result.get('title', 'N/A')}\n"
                            investment_opportunities += f"URL: {result.get('url', 'N/A')}\n"
                            investment_opportunities += f"Content: {result.get('content', 'N/A')}\n"
                else:
                    # For non-list results, use the string representation
                    result_str = str(results)
                    if result_str not in seen_results:
                        seen_results.add(result_str)
                        investment_opportunities += f"\nSearch: {prompt}\n"
                        investment_opportunities += f"Results:\n{results}\n"
            except Exception as e:
                st.warning(f"Error in search for '{prompt}': {str(e)}")
                continue
        
        # Get conversation history
        conversation_history = st.session_state.current_conversation
        
        # After collecting investment opportunities
        print(investment_opportunities)
        
        # Include conversation history in the advice prompt
        conversation_context = "\nPrevious Conversation:\n"
        for message in conversation_history[-5:]:  # Last 5 messages for context
            conversation_context += f"{message['role']}: {message['content']}\n"
        
        advice_prompt = f"""
        You are a financial advisor. Provide personalized financial advice based on the following information:
        
        User Query: {query}
        
        User Profile:
        - Name: {user_profile['name']}
        - Age: {user_profile['age']}
        - Income: {user_profile['income']}
        - Location: {user_profile['city']}, {user_profile['country']}
        - Financial Goals: {', '.join(user_profile['financial_goals'])}
        - Risk Tolerance: {user_profile['risk_tolerance']}
        - Investment Preferences: {', '.join(user_profile['investment_preferences'])}
        
        Transaction History Summary:
        - Total Transactions: {len(df) if df is not None else 0}
        - Total Spent: ${total_spent:,.2f}
        - Average Transaction: ${avg_transaction:,.2f}
        - Top Categories: {top_categories}
        
        
        Current Investment Opportunities:
        {investment_opportunities}
        

        
        Please provide:
        1. Personalized financial advice based on the user's profile and spending patterns
        2. Location-specific recommendations for financial products or services
        3. Actionable steps to achieve their financial goals considering local context
        4. Risk-appropriate investment suggestions for their region
        5. Analysis of current investment opportunities in relation to their profile and location
        6. Market trends and economic factors specific to their city and country
        7. Long-term investment strategy recommendations considering local regulations
        8. Tax implications and benefits specific to their location
        
        
        Previous Conversation:
        {conversation_context}
        
        """
        
        print(advice_prompt)
        
        response_placeholder = st.empty()
        full_response = "Financial Advice:\n"
        for chunk in llm.stream(advice_prompt):
            if chunk:
                full_response += chunk
                response_placeholder.markdown(full_response)
        
        # Save the investment opportunities and final advice with conversation history
        save_investment_opportunities(investment_opportunities, user_profile, query, full_response, conversation_history)
                
        return full_response
            
    except Exception as e:
        return f"Error generating financial advice: {str(e)}"

def get_investment_opportunities(query, user_profile):
    """Search for investment opportunities based on user profile and query."""
    try:
        # Create specialized search prompts based on user profile
        search_prompts = [
            f"current investment opportunities for {user_profile['risk_tolerance']} risk tolerance investors",
            f"investment strategies for {user_profile['age']} year old with {user_profile['income']} income",
            f"best {user_profile['investment_preferences']} investments 2024",
            f"how to achieve {user_profile['financial_goals']} through investments"
        ]
        
        # Initialize search tool
        search = DuckDuckGoSearchRun()
        
        # Search for each prompt and combine results
        full_response = "Investment Opportunities and Research:\n\n"
        for prompt in search_prompts:
            results = search.run(prompt)
            full_response += f"Search: {prompt}\n"
            full_response += f"Results:\n{results}\n\n"
        
        return full_response
    except Exception as e:
        return f"Error searching for investment opportunities: {str(e)}"

def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, NewConnectionError):
        return False

class AgentCallbackHandler(BaseCallbackHandler):
    def __init__(self, verbose_placeholder):
        self.verbose_placeholder = verbose_placeholder
        self.thoughts = ""
    
    def on_agent_action(self, action, **kwargs):
        self.thoughts += f"\nThought: {action.log}\n"
        self.verbose_placeholder.markdown(f"```\n{self.thoughts}\n```")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.thoughts += f"\nAction: Using tool {serialized['name']}\nInput: {input_str}\n"
        self.verbose_placeholder.markdown(f"```\n{self.thoughts}\n```")
    
    def on_tool_end(self, output, **kwargs):
        self.thoughts += f"\nObservation: {output}\n"
        self.verbose_placeholder.markdown(f"```\n{self.thoughts}\n```")
    
    def on_agent_finish(self, finish, **kwargs):
        self.thoughts += f"\nFinal Answer: {finish.return_values['output']}\n"
        self.verbose_placeholder.markdown(f"```\n{self.thoughts}\n```")

def create_agent():
    """Create the agent with LangGraph."""
    # Check if Ollama is running
    if not check_ollama_connection():
        st.error("Ollama is not running. Please start Ollama and ensure it's accessible at http://localhost:11434")
        return None

    try:
        # Define the tools
        tools = [
            Tool(
                name="get_account_summary",
                description="Get account summary including balances and transaction history",
                func=lambda _: get_account_summary(st.session_state.current_query,
                                                 st.session_state.user_transactions)
            ),
            Tool(
                name="get_financial_advice",
                description="Get personalized financial advice based on user profile and transaction history",
                func=lambda _: get_financial_advice(
                    st.session_state.current_query,
                    st.session_state.user_transactions,
                    st.session_state.user_profile
                )
            )
        ]
        
        # Initialize memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create callback handler
        verbose_placeholder = st.empty()
        callback_handler = AgentCallbackHandler(verbose_placeholder)
        
        # Initialize the agent
        agent = initialize_agent(
            tools,
            llm_tool,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[callback_handler]
        )
        
        return agent
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

def get_agent_response(query):
    """Get response from the agent using transaction data and policy documents."""
    if st.session_state.transaction_data is None:
        st.session_state.transaction_data = load_transaction_data()
    
    if st.session_state.transaction_data is None:
        return {"role": "assistant", "content": "Unable to load transaction data. Please try again later."}
    
    # Save the current query in session state
    st.session_state.current_query = query
    
    # Get user transactions if not already loaded
    if st.session_state.user_transactions is None:
        st.session_state.user_transactions = get_user_transactions(
            st.session_state.transaction_data,
            st.session_state.user_profile['first_name'],
            st.session_state.user_profile['last_name'],
            st.session_state.user_profile['city'],
            st.session_state.user_profile['country']
        )
    
    # Initialize the agent if not already done
    if "agent" not in st.session_state:
        st.session_state.agent = create_agent()
    
    if st.session_state.agent is None:
        return {"role": "assistant", "content": "Unable to initialize the agent. Please check if Ollama is running."}
    
    try:
        # Add user message to conversation
        st.session_state.current_conversation.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Create placeholders for both verbose output and main response
        verbose_placeholder = st.empty()
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in st.session_state.agent.stream(query):
            if isinstance(chunk, dict):
                # Handle verbose output through callbacks
                if "intermediate_steps" in chunk:
                    for step in chunk["intermediate_steps"]:
                        if isinstance(step, tuple) and len(step) == 2:
                            action, observation = step
                            # The callback handler will update the verbose output
                            # We'll focus on the final response in the main chat
                # Handle final response
                if "output" in chunk:
                    full_response += chunk["output"]
                    response_placeholder.markdown(full_response)
            elif isinstance(chunk, str):
                full_response += chunk
                response_placeholder.markdown(full_response)
        
        # Add the complete response to conversation
        st.session_state.current_conversation.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return {"role": "assistant", "content": full_response}
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return {"role": "assistant", "content": "I apologize, but I encountered an error while processing your request. Please try again."}

def reset_conversation():
    """Reset the current conversation and save it to history."""
    if st.session_state.current_conversation:
        st.session_state.conversation_history.append({
            "id": len(st.session_state.conversation_history) + 1,
            "conversation": st.session_state.current_conversation.copy(),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    st.session_state.current_conversation = []
    if "agent" in st.session_state:
        del st.session_state.agent

def update_user_profile():
    """Update user profile based on form input."""
    with st.form("user_profile_form"):
        st.header("Update Your Profile")
        name = st.text_input("Name", value=st.session_state.user_profile["name"])
        first_name = st.text_input("First Name", value=st.session_state.user_profile["first_name"])
        last_name = st.text_input("Last Name", value=st.session_state.user_profile["last_name"])
        city = st.text_input("City", value=st.session_state.user_profile["city"])
        country = st.text_input("Country", value=st.session_state.user_profile["country"])
        age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.user_profile["age"])
        income = st.number_input("Annual Income", min_value=0, value=st.session_state.user_profile["income"])
        financial_goals = st.multiselect(
            "Financial Goals",
            ["Retirement", "Home Purchase", "Education", "Investment", "Debt Reduction", "Emergency Fund"],
            default=st.session_state.user_profile["financial_goals"]
        )
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["low", "medium", "high"],
            value=st.session_state.user_profile["risk_tolerance"]
        )
        investment_preferences = st.multiselect(
            "Investment Preferences",
            ["Stocks", "Bonds", "Mutual Funds", "ETFs", "Real Estate", "Cryptocurrency"],
            default=st.session_state.user_profile["investment_preferences"]
        )
        
        if st.form_submit_button("Update Profile"):
            # Update session state
            st.session_state.user_profile = {
                "name": name,
                "first_name": first_name,
                "last_name": last_name,
                "city": city,
                "country": country,
                "age": age,
                "income": income,
                "financial_goals": financial_goals,
                "risk_tolerance": risk_tolerance,
                "investment_preferences": investment_preferences
            }
            
            # Update config file
            config = load_credentials()
            if st.session_state.current_user in config["users"]:
                config["users"][st.session_state.current_user]["profile"] = st.session_state.user_profile
                save_credentials(config)
            
            # Reload user transactions with new profile
            if st.session_state.transaction_data is not None:
                st.session_state.user_transactions = get_user_transactions(
                    st.session_state.transaction_data,
                    first_name,
                    last_name,
                    city,
                    country
                )
            st.success("Profile updated successfully!")

# Main app
def main():
    st.title("Financial Assistant")
    
    # Authentication
    if not st.session_state.get("password_correct", False):
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            login()
        with tab2:
            if signup():
                st.rerun()
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Chat", "Profile", "History"])
    
    with tab1:
        # Reset button
        if st.button("Reset Conversation"):
            reset_conversation()
            st.rerun()
        
        # Policy Documents Upload
        st.header("Financial Documents")
        uploaded_files = st.file_uploader("Upload financial documents (PDF)", accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Processing documents..."):
                st.session_state.policy_store = process_policy_documents(uploaded_files)
                if st.session_state.policy_store:
                    st.success("Documents processed successfully!")
        
        # Input section
        st.header("Financial Assistant")
        query = st.text_input("Ask about your accounts, get financial advice, or request support (e.g., 'Show my account summary', 'What investments should I consider?', 'Help me plan for retirement'):")
        
        # Display current conversation
        st.header("Current Conversation")
        for message in st.session_state.current_conversation:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                st.caption(message["timestamp"])
        
        # Query processing
        if st.button("Submit Query"):
            if query:
                with st.spinner("Processing your request..."):
                    response = get_agent_response(query)
                    if response:
                        with st.chat_message(response["role"]):
                            st.markdown(response["content"], unsafe_allow_html=True)
                            st.caption(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                st.warning("Please enter a query.")
    
    with tab2:
        update_user_profile()
    
    with tab3:
        st.header("Conversation History")
        for conversation in reversed(st.session_state.conversation_history):
            with st.expander(f"Conversation {conversation['id']} - {conversation['timestamp']}"):
                for message in conversation["conversation"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)
                        st.caption(message["timestamp"])

main() 