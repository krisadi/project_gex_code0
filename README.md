# Financial Assistant Application

A comprehensive financial advisory system that provides personalized financial advice, investment opportunities, and account management based on user profiles and transaction history.
Dataset: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset



## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Components](#components)
5. [Setup](#setup)
6. [Usage](#usage)
7. [Security](#security)
8. [Data Storage](#data-storage)
9. [API Integration](#api-integration)

## Overview

The Financial Assistant is a Streamlit-based web application that combines AI-powered financial advice with user profile management and transaction analysis. It uses LangChain, Ollama, and Tavily Search to provide personalized financial recommendations.

## Features

### User Management
- Secure login and signup system
- User profile management
- Session-based authentication
- Profile persistence using YAML configuration

### Financial Analysis
- Transaction history analysis
- Account summary generation
- Spending pattern analysis
- Category-based expense tracking

### Investment Research
- Location-specific investment opportunities
- Risk tolerance-based recommendations
- Market trend analysis
- Tax implications and benefits

### AI-Powered Advice
- Personalized financial recommendations
- Context-aware responses
- Conversation history tracking
- Real-time streaming responses

## Architecture

### Core Components
1. **User Interface (Streamlit)**
   - Interactive web interface
   - Real-time chat interface
   - Profile management forms
   - Document upload system

2. **AI Engine**
   - LangChain for agent management
   - Ollama for LLM processing
   - Tavily Search for investment research
   - Custom callback handlers

3. **Data Management**
   - YAML-based configuration
   - Transaction data processing
   - Investment records storage
   - Conversation history tracking

### Data Flow
1. User input → Streamlit interface
2. Query processing → LangChain agent
3. Tool execution → External APIs
4. Response generation → LLM
5. Result storage → Local files
6. Response display → User interface

## Components

### Main Application (`main.py`)
- Streamlit application setup
- User authentication
- Profile management
- Financial advice generation
- Investment opportunity search
- Conversation handling

### Server Component (`mcp_server.py`)
- FastAPI server implementation
- API key validation
- Query processing
- Context-aware responses

### Configuration (`config.yaml`)
- User credentials
- Profile information
- System settings

## Setup

### Prerequisites
- Python 3.8+
- Ollama running locally
- Tavily API key
- Required Python packages (see requirements.txt)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` file with API keys:
   ```
   TAVILY_API_KEY=your_api_key
   ```
4. Start Ollama service
5. Run the application:
   ```bash
   streamlit run main.py
   ```
6. setup using uv and run:
    ```bash
    uv run python -m streamlit run main.py
    ```

## Usage

### User Authentication
1. Sign up with profile information
2. Log in with credentials
3. Update profile as needed

### Financial Advice
1. Enter queries in the chat interface
2. Upload financial documents
3. View personalized recommendations
4. Track conversation history

### Investment Research
1. Search for opportunities
2. View location-specific results
3. Analyze market trends
4. Get personalized recommendations

## Security

### Authentication
- Password hashing using SHA-256
- Session-based authentication
- API key validation
- Secure credential storage

### Data Protection
- Environment variable management
- Secure file handling
- Input validation
- Error handling

## Data Storage

### User Data
- Stored in `config.yaml`
- Encrypted passwords
- Profile information
- Session data

### Investment Records
- Stored in `investment_records/`
- JSON format
- Timestamp-based naming
- Complete context storage

### Transaction Data
- CSV file processing
- In-memory caching
- Real-time analysis
- Historical tracking
- create a folder data/###.csv (Download from https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)

## API Integration

### Tavily Search
- Investment research
- Market analysis
- Location-specific queries
- Real-time data retrieval

### Ollama
- LLM processing
- Response generation
- Context management
- Streaming support

### FastAPI Server
- Query processing
- Context handling
- Response formatting
- API key validation
