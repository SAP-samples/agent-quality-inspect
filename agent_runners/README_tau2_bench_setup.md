# Tau2 Bench Agent

**Paper:** [Tau2 Bench (arXiv)](https://arxiv.org/pdf/2408.04682v1)  
**Original Repo:** [GitHub - sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)

---

## Installation

1. **Install**
    ```bash
    cd <parent_dir>/agent_runners/tau2-bench
    python3.10 -m venv .tau2benchenv
    source .tau2benchenv/bin/activate
    pip install -e .
    pip install fastapi
    ```

2. **Configure API Keys**
    - Rename `.env copy` to `.env`.
    - Fill in required API keys.
    - For non-Azure providers, update both `.env` and `config.py` accordingly.
    - For Azure make sure the endpoint is from Azure Studio, i.e. it doesn't end with /models.

3. **Start the API Server**
    ```bash
    uvicorn fastapi_server.main:app --reload --port 5500
    ```

---

## API Endpoints

| Endpoint                                      | Method | Description                              |
|------------------------------------------------|--------|------------------------------------------|
| `/start_conversation`                         | POST   | Start a new conversation session         |
| `/conversation/{session_id}/messages`         | POST   | Send a message in an existing session    |
| `/conversation/{session_id}/end`              | POST   | End a conversation session               |
| `/conversation/{session_id}/trajectory`       | GET    | Get the full conversation trajectory     |
| `/agent_description`                          | POST   | Get the agent description for a particular domain |

---
### Start Conversation
**Endpoint:** `/start_conversation`  
**Method:** `POST`  
**Description:** Starts a new conversation session.

**Request Body:**
```json
{
    "domain": "airline",
    "agent_llm": "azure/gpt-4.1"
}
```
**Response:**
```json
{
    "session_id": "unique_session_id"
}
```

---

### Continue Conversation
**Endpoint:** `/conversation/{session_id}/messages`  
**Method:** `POST`  
**Description:** Continues the conversation with a new message. Using the session_id from the start conversation response.

**Request Body:**
```json
{
    "message": "Your message here"
}
```
**Response:**
```json
{
    "response": "Agent's response here",
    "history": [
        {
            "role": "user",
            "content": "Your message here"
        },
        {
            "role": "agent",
            "content": "Agent's response here"
        }
    ]
}
```

---

### End Conversation
**Endpoint:** `/conversation/{session_id}/end`  
**Method:** `POST`  
**Description:** Ends the conversation session. Using the session_id from the start conversation response.  
**Note/Warning:** This will end the session so you cannot extract the trajectory after this.

**Response:**
```json
{
    "status": "ended"
}
```

---

### Get Trajectory
**Endpoint:** `/conversation/{session_id}/trajectory`  
**Method:** `GET`  
**Description:** Retrieves the trajectory of the conversation session. Using the session_id from the start conversation response.

**Response:**
```json
{
    "trajectory": [
        [
            {
                "role": "user",
                "content": "Turn 1 Your message here"
            },
            {
                "role": "agent",
                "content": "Turn 1 Agent's response here",
                "tool_calls": "Turn 1 Agent tool calls here"
            }
        ],
        [
            {
                "role": "user",
                "content": "Turn 2 Your message here"
            },
            {
                "role": "agent",
                "content": "Turn 2 Agent's response here",
                "tool_calls": "Turn 2 Agent tool calls here"
            }
        ]
    ]
}
```

---

### End Conversation
**Endpoint:** `/agent_description`  
**Method:** `POST`  
**Description:** Retrieves the agent description for a particular domain. Reads the policy.md file in the data folder.

**Request Body:**
```json
{
    "domain": ["airline", "retail", "telecom"]
}
```

**Response:**
```json
{
    "description": "agent description"
}
```
