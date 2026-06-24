from abc import ABC, abstractmethod
from typing import Dict, Any
import json

from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace


class BaseAdapter(ABC):
    """
    Abstract base class for converting external agent trace formats to AgentDialogueTrace format.
    """

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """
        Load JSON data from file.

        :param file_path: Path to the JSON file to load.
        :return: Loaded JSON data as a dictionary.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @abstractmethod
    def convert_to_agent_trace(self, conversation_data: Any) -> AgentDialogueTrace:
        """
        Convert external conversation format to AgentDialogueTrace format.

        :param conversation_data: Conversation data in external format (format varies by adapter).
        :return: Converted agent trace.
        """
        pass
