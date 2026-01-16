import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# CHANGED: We now import Google's class instead of OpenAI's
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 1. Load the GOOGLE_API_KEY from .env
load_dotenv()

app = FastAPI(title="Day 1: AI Data Extractor (Free Tier)")

# 2. Define Output Structure (Same as before)
class TopicSummary(BaseModel):
    topic: str = Field(description="The topic provided by the user")
    summary: str = Field(description="A concise summary in 2 sentences")
    sentiment: str = Field(description="General sentiment (Positive, Negative, Neutral)")
    keywords: List[str] = Field(description="3-5 technical keywords")
    difficulty_level: int = Field(description="Score 1-10 on difficulty")

# 3. Define Input Request
class RequestBody(BaseModel):
    topic: str

# 4. Core Logic
def get_ai_summary(topic_input: str) -> TopicSummary:
    # CHANGED: Initialize Google Gemini model
    # "gemini-2.5-flash" is the best model for the free tier
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    parser = PydanticOutputParser(pydantic_object=TopicSummary)

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical writer. Return ONLY JSON.\n{format_instructions}"),
        ("user", "Topic: {topic}")
    ])

    # Chain
    chain = prompt | llm | parser

    # Invoke
    result = chain.invoke({
        "topic": topic_input,
        "format_instructions": parser.get_format_instructions()
    })
    
    return result

# 5. API Endpoint
@app.post("/summarize", response_model=TopicSummary)
async def summarize_topic(request: RequestBody):
    try:
        result = get_ai_summary(request.topic)
        return result
    except Exception as e:
        # If Gemini refuses to output JSON, this catches the error
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)