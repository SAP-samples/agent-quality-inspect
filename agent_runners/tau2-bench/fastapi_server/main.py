from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, List, Optional
import importlib
import os

# Import tau2 modules (assuming installed in editable mode or PYTHONPATH set)
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import UserMessage, AssistantMessage, ToolMessage
from tau2.domains.airline.environment import get_environment

app = FastAPI()

# In-memory session store
dialog_sessions: Dict[str, dict] = {}

class StartConversationRequest(BaseModel):
    domain: str = "airline"
    agent_llm: str = "azure/gpt-4.1"
    # Optionally, add task_id, etc.
    # task_id: Optional[str] = None

class StartConversationResponse(BaseModel):
    session_id: str

class UserMessageRequest(BaseModel):
    message: str

class AgentResponse(BaseModel):
    response: str
    history: List[dict]

class AgentDescriptionRequest(BaseModel):
    domain: str

@app.post("/start_conversation", response_model=StartConversationResponse)
def start_conversation(req: StartConversationRequest):
    # Dynamically import the environment module for the requested domain
    try:
        env_module = importlib.import_module(f"tau2.domains.{req.domain}.environment")
        get_env_func = getattr(env_module, "get_environment")
        env = get_env_func()
    except Exception as e:
        raise HTTPException(400, f"Could not load environment for domain '{req.domain}': {e}")
    tools = list(env.tools.get_tools().values())
    agent = LLMAgent(llm=req.agent_llm, tools=tools, domain_policy=env.policy)
    agent_state = agent.get_init_state()
    session_id = str(uuid4())
    dialog_sessions[session_id] = {
        "agent": agent,
        "agent_state": agent_state,
        "environment": env,
        "history": [],
    }
    return {"session_id": session_id}

@app.post("/conversation/{session_id}/message", response_model=AgentResponse)
def send_message(session_id: str, req: UserMessageRequest):
    session = dialog_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    # Create UserMessage and append to history
    user_msg = UserMessage(role="user", content=req.message)
    session["history"].append({"role": "user", "content": req.message})
    agent = session["agent"]
    agent_state = session["agent_state"]
    env = session["environment"]
    # Agent step: loop until agent produces a user-facing message (not a tool call)
    next_input = user_msg
    while True:
        print(f"[DEBUG] Making agent call...)")
        agent_msg, agent_state = agent.generate_next_message(next_input, agent_state)
        print(f"[DEBUG] Agent call complete")
        session["history"].append({"role": "agent", "content": agent_msg.content, "tool_calls": getattr(agent_msg, "tool_calls", None)})
        if agent_msg.is_tool_call() and agent_msg.tool_calls:
            tool_msgs = []
            for tool_call in agent_msg.tool_calls:
                tool_msg = env.get_response(tool_call)
                tool_msgs.append(tool_msg)
                session["history"].append({"role": "tool", "content": tool_msg.content, "tool_id": tool_msg.id})
            # If multiple tool messages, wrap in MultiToolMessage, else just use the single one
            if len(tool_msgs) == 1:
                next_input = tool_msgs[0]
            else:
                from tau2.data_model.message import MultiToolMessage
                next_input = MultiToolMessage(role="tool", tool_messages=tool_msgs)
            continue
        else:
            break
    session["agent_state"] = agent_state
    return {"response": agent_msg.content, "history": session["history"]}

@app.post("/conversation/{session_id}/end")
def end_conversation(session_id: str):
    if session_id in dialog_sessions:
        del dialog_sessions[session_id]
        return {"status": "ended"}
    else:
        raise HTTPException(404, "Session not found")

@app.get("/conversation/{session_id}/trajectory")
def get_trajectory(session_id: str):
    session = dialog_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    history = session["history"]
    turns = []
    current_turn = []
    for msg in history:
        if msg["role"] == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)
    if current_turn:
        turns.append(current_turn)
    return {"trajectory": turns}

@app.post("/agent_description")
def get_agent_description(req: AgentDescriptionRequest):
    # Compose the path to the policy.md file for the given domain

    base_path = os.path.join(os.path.dirname(__file__), "..", "data", "tau2", "domains")
    policy_path = os.path.abspath(os.path.join(base_path, req.domain, "policy.md"))
    if not os.path.isfile(policy_path):
        raise HTTPException(404, f"Policy file not found for domain '{req.domain}'")
    with open(policy_path, "r") as f:
        description = f.read()
    return {"description": description}