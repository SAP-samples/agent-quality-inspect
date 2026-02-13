from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ResponseFromAgent:
    """
    Represents agent response and other related outputs.
    """

    response_str: str
    """
    Agent response.
    """
    json_data: Optional[dict] = None
    """
    Additional outputs by the agent.
    """


@dataclass
class UserProxyMessage:
    """
    Represents an utterance from the user proxy.
    """

    message_str: str 
    """
    User proxy utterance. May contain stop sequence if the conversation terminates early.
    """
    check: Optional[str] = None
    """
    Stores the terminating condition that is applied when the user proxy exits the conversation early.
    """


@dataclass
class ConversationTurn:
    """
    Represents a single conversation exchange between user and agent in the current conversational turn.
    """

    id: str
    """
    The unique identifier for the current conversation turn.
    """
    agent_responses: List[ResponseFromAgent]
    """
    Agent outputs, which may include responses and other related outputs stored in a list.
    """
    user_message : UserProxyMessage
    """
    User proxy output.
    """


@dataclass
class ChatHistory:
    """
    Represents the user-agent conversation history.
    """

    id: str
    """
    The unique identifier for the chat history.
    """
    conversations: List[ConversationTurn]
    """
    A list of user–agent exchanges across conversational turns.
    """
