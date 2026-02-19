from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.models.user_proxy.chat import ChatHistory, UserProxyMessage


# add expert non expert config

class LLMProxyAgent(ABC):
    """
    Abstract class which should be extended for actual implementation of LLM agent.

    :param llm_client: the connection to the llm client for response generation.
    :param config: configuration for LLM agent initialization. Default to ``None``.
    """
    def __init__(
            self,
            llm_client: LLMClient,
            config: Optional[Dict[str, Any]] = None
    ):
        self.llm_client = llm_client
        self.config = config or {}

    @abstractmethod
    async def generate_message_from_chat_history(self, chat_history: ChatHistory) -> UserProxyMessage:
        """
        This is an abstract method and should be implemented in a concrete class.

        :param chat_history: a :obj:`~agent_inspect.models.user_proxy.chat.ChatHistory` object containing the conversation history. 
        :return: a :obj:`~agent_inspect.models.user_proxy.chat.UserProxyMessage` object containing the LLM agent response.
        """
        ...
