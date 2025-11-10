from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
REGION_NAME = # Insert your AWS region name here
CREDENTIALS_PROFILE_NAME = # Insert your AWS credentials profile name here

EMBEDDER_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDER_MODEL_KWARGS = {
    "dimensions": 1024,
    "normalize": True
}

LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
LLM_MODEL_KWARGS = {
    "max_tokens": 1024,
    "temperature": 0.1
}

SEARCH_TYPE = "similarity"
RETRIEVER_KWARGS = {
    "k": 5 # Number of documents to retrieve
}

VECTOR_STORE_PATH = "./vector_database/"

PROMPT_TEMPLATE = """
You are a Friends TV show episode assistant. Based on the episode information provided, give a clear and concise response.
Don't make up any information. If the episode is not found in the context, respond with "Episode not found in the provided information."

Episode Information:
{context}

User Query: {question}

Please respond with the season number, episode number within the season, episode number of the entire series, title, and a brief description of the episode. 
Don't forget to mention if the information is based on episode summaries or script scenes.

RAG Answer:"""

# Initialize embedder
embedder = BedrockEmbeddings(
    model_id=EMBEDDER_MODEL_ID,
    model_kwargs=EMBEDDER_MODEL_KWARGS,
    region_name=REGION_NAME,
    credentials_profile_name=CREDENTIALS_PROFILE_NAME
)

# Initialize LLM
llm = ChatBedrock(
        region_name=REGION_NAME, 
        credentials_profile_name=CREDENTIALS_PROFILE_NAME,
        model_id=LLM_MODEL_ID, 
        model_kwargs=LLM_MODEL_KWARGS
    )

# Load vector store and create retriever
vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings=embedder, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs=RETRIEVER_KWARGS)

# Create RAG chain
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Initialize FastAPI app
app = FastAPI()

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Query about Friends TV show episodes")

class Document(BaseModel):
    content: str = Field(..., description="Content of the document")
    metadata: dict = Field(..., description="Metadata associated with the document")

class QueryResponse(BaseModel):
    response: str = Field(..., description="Response from the Friends RAG system")
    retrieved_documents: list[Document] = Field(..., description="List of documents retrieved to answer the query")

@app.get("/")
def root():
    return {"message": "Friends RAG API is running!"}

@app.post("/query", response_model=QueryResponse)
def query_episodes(request: QueryRequest):
    """Query the Friends RAG system for episode information"""
    try:
        # Get the RAG response
        response = rag_chain.invoke(request.query)
        
        # Get the retrieved documents
        docs = retriever.invoke(request.query)
        
        # Format documents for response
        retrieved_docs = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return {
            "response": response,
            "retrieved_documents": retrieved_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
