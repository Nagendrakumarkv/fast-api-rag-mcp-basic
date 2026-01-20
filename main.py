import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Annotated, List, TypedDict
import operator

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

# 1. Load Env
load_dotenv()

app = FastAPI(title="Day 6: Hybrid Search Agent")

# --- HYBRID SEARCH SETUP ---

print("⚙️  Initializing Hybrid Search...")

# A. Setup Pinecone (Dense / Semantic Search)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = PineconeVectorStore(index_name="rag-app", embedding=embeddings)
pinecone_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# B. Setup BM25 (Sparse / Keyword Search)
# We load the PDF again to build the keyword index in RAM
loader = PyPDFLoader("my_cv.pdf") 
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 3 # Get top 3 keyword matches

def ensemble_search(query: str, bm25, vector, k: int = 3):
    """
    Hybrid search combining BM25 (keyword) and Vector (semantic).
    Works on ALL LangChain versions.
    """
    bm25_docs = bm25.get_relevant_documents(query)
    vector_docs = vector.get_relevant_documents(query)

    # Deduplicate by page_content
    seen = set()
    final_docs = []

    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            final_docs.append(doc)

    return final_docs[:k]


print("✅ Hybrid Search Ready!")

# --- AGENT TOOLS ---

@tool
def search_hybrid_knowledge_base(query: str):
    """Use this tool to find specific details, IDs, names, or general concepts in the candidate's document."""
    print(f"🕵️‍♂️ HYBRID SEARCH: '{query}'")
    
    # This runs BOTH Pinecone and BM25, then combines results
    docs = ensemble_search(
    query=query,
    bm25=bm25_retriever,
    vector=pinecone_retriever)
    
    return "\n\n".join([doc.page_content for doc in docs])

tools = [search_hybrid_knowledge_base]

# --- GRAPH SETUP (Same as Day 5) ---

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

def chatbot_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    return "tools" if last_msg.tool_calls else "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app_graph = workflow.compile()

# --- API ENDPOINT ---
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/agent", response_model=ChatResponse)
async def run_hybrid_agent(request: ChatRequest):
    try:
        inputs = {"messages": [HumanMessage(content=request.question)]}
        final_state = app_graph.invoke(inputs)
        
        # Safe String Conversion (from Day 5 Fix)
        raw = final_state["messages"][-1].content
        if isinstance(raw, list):
            final = "".join([p.get("text","") for p in raw if "text" in p])
        else:
            final = str(raw)
            
        return ChatResponse(answer=final)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)