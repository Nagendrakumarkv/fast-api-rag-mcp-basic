import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Annotated, List

# LangChain & Graph Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing_extensions import TypedDict
import operator

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="Day 5: LangGraph AI Agent")

# --- CONFIGURATION ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = PineconeVectorStore(index_name="rag-app", embedding=embeddings)

# --- DEFINE TOOLS ---

@tool
def search_pdf_knowledge_base(query: str):
    """Use this tool when answering questions about the candidate's resume, CV, or specific document info."""
    print(f"🕵️‍♂️ AGENT LOG: Searching PDF for '{query}'...")
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def search_web(query: str):
    """Use this tool for current events, weather, or general knowledge NOT in the resume."""
    print(f"🌍 AGENT LOG: Searching Web for '{query}'...")
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

# List of tools the AI can use
tools = [search_pdf_knowledge_base, search_web]

# --- SETUP MODEL WITH TOOLS ---
# We bind the tools to the model so it knows they exist
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- GRAPH STATE ---
# This dictionary holds the conversation history
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

# --- GRAPH NODES ---

def chatbot_node(state: AgentState):
    """The 'Brain' node: It looks at history and decides what to do next."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- GRAPH EDGES (The Router) ---

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Decides: Did the AI ask to use a tool? OR is it done talking?"""
    last_message = state["messages"][-1]
    
    # If the AI produced a 'tool_calls' attribute, go to the 'tools' node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, stop
    return "__end__"

# --- BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

# 1. Add Nodes
workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", ToolNode(tools)) # Prebuilt node that executes tools

# 2. Add Edges
workflow.set_entry_point("agent") # Start here
workflow.add_conditional_edges("agent", should_continue) # Decide where to go
workflow.add_edge("tools", "agent") # Loop back to agent after using a tool

# 3. Compile
app_graph = workflow.compile()

# --- API ENDPOINT ---
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default" # Simplified for Day 5 demo

class ChatResponse(BaseModel):
    answer: str

@app.post("/agent", response_model=ChatResponse)
async def run_agent(request: ChatRequest):
    try:
        # Run the graph
        inputs = {"messages": [HumanMessage(content=request.question)]}
        final_state = app_graph.invoke(inputs)
        
        # Get the last message
        last_message = final_state["messages"][-1]
        raw_content = last_message.content
        
        # --- THE FIX: Handle List vs String content ---
        final_answer = ""
        
        if isinstance(raw_content, str):
            # If it's already a string, just use it
            final_answer = raw_content
        elif isinstance(raw_content, list):
            # If it's a list (Google sometimes does this), join the text parts
            # Example: [{'type': 'text', 'text': 'He knows Python'}]
            final_answer = "".join(
                [part.get("text", "") for part in raw_content if "text" in part]
            )
        else:
            # Fallback for unexpected types
            final_answer = str(raw_content)
            
        return ChatResponse(answer=final_answer)
        
    except Exception as e:
        print(f"Error: {e}")
        # Print the actual content causing the error to help debug
        if 'final_state' in locals():
            print(f"Failed Content Structure: {final_state['messages'][-1].content}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)