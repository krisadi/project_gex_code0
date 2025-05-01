from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

app = FastAPI(title="MCP Server")

# Initialize components
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# API Key validation
def verify_api_key(api_key: str = Header(None)):
    if api_key != os.getenv("MCP_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        # Process the query using the provided context
        if request.context:
            # Create a vector store from the context
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.create_documents([request.context])
            vector_store = FAISS.from_documents(docs, embeddings)
            
            # Create a QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )
            
            # Get the answer
            answer = qa_chain.run(request.query)
            sources = ["Provided context"]
        else:
            # If no context, use the LLM directly
            answer = llm.invoke(request.query).content
            sources = ["Direct LLM response"]
            
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 