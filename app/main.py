from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent import make_agent
from app.models import ChatRequest, ChatResponse
from app.rag import build_vectorstore

sessions: Dict[str, List[BaseMessage]] = {}
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    vectorstore = build_vectorstore()
    agent = make_agent(vectorstore)
    yield


app = FastAPI(
    title="CdekStart Internship Chatbot",
    description="RAG-powered chatbot for SDEK international internship program",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session_id = request.session_id
    history = sessions.get(session_id, [])

    state = {
        "query": request.message,
        "history": history,
        "retrieved_docs": [],
        "needs_clarification": False,
        "response": "",
    }

    result = agent.invoke(state)
    response_text = result["response"]

    updated_history = history + [
        HumanMessage(content=request.message),
        AIMessage(content=response_text),
    ]
    sessions[session_id] = updated_history

    return ChatResponse(response=response_text, session_id=session_id)


@app.get("/health")
async def health():
    return {"status": "ok"}
