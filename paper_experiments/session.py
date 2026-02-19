from dataclasses import dataclass
import logging
import requests
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from dotenv import load_dotenv

@dataclass
class SessionMessageResponse:
    """
    Represents the response received after sending a message in a session.
    """

    response_message: str
    """
    The plaintextmessage content returned by the agent.
    """

class BaseSession(ABC):
    """
    Base class for managing API sessions.
    
    To extend, implement:
    1. get_config() - Return configuration dict with endpoints and base_url
    2. Optionally override build_payload() methods for custom payloads
    """

    def __init__(self, identifier: str, agent_type: str):
        load_dotenv()
        self.identifier = identifier
        self.agent_type = agent_type
        self.session_id: Optional[str] = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Get configuration once during initialization
        config = self.get_config()
        self.base_url = config["base_url"]
        self.endpoints = config["endpoints"]
        
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary.
        
        Required keys:
        - base_url: str
        - endpoints: dict with keys: start, message, trajectory, end
        
        Example:
        {
            "base_url": "http://localhost:5500",
            "endpoints": {
                "start": "/start_conversation",
                "message": "/conversation/{session_id}/message",
                "trajectory": "/conversation/{session_id}/trajectory",
                "end": "/conversation/{session_id}/end"
            }
        }
        """
        pass
    
    def build_start_payload(self) -> Dict[str, Any]:
        """Override to customize start session payload."""
        return {
            "agent_type": self.agent_type,
            "identifier": self.identifier
        }
    
    def build_message_payload(self, message: str) -> Dict[str, Any]:
        """Override to customize message payload."""
        return {"message": message}
    
    def process_trajectory(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override to customize trajectory processing."""
        return response_data
    
    def start_session(self) -> str:
        """Start a new session and store the session_id."""
        url = f"{self.base_url}{self.endpoints['start']}"
        payload = self.build_start_payload()
        
        self.logger.info(f"Starting session for {self.identifier}")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            self.session_id = data.get("session_id")
            self.logger.info(f"Session started with ID: {self.session_id}")
            return self.session_id
        else:
            error_msg = f"Failed to start session: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
    def _ensure_session_active(self) -> None:
        if not self.session_id:
            raise Exception("Session is not active. Please start a session first.")
    
    @abstractmethod
    def send_message(self, message: str) -> SessionMessageResponse:
        """Send a message to the current session."""
        ...
        
    def get_trajectory(self) -> Dict[str, Any]:
        """Retrieve the trajectory for current session."""
        self._ensure_session_active()
        
        url = f"{self.base_url}{self.endpoints['trajectory'].format(session_id=self.session_id)}"
        self.logger.info(f"Retrieving trajectory for session {self.session_id}")
        response = requests.get(url)
        
        if response.status_code == 200:
            return self.process_trajectory(response.json())
        else:
            error_msg = f"Failed to retrieve trajectory: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
    def end(self) -> None:
        """End the current session."""
        self._ensure_session_active()
        
        url = f"{self.base_url}{self.endpoints['end'].format(session_id=self.session_id)}"
        self.logger.info(f"Ending session {self.session_id}")
        response = requests.post(url)
        
        if response.status_code == 200:
            self.logger.info(f"Session {self.session_id} ended successfully.")
            self.session_id = None
        else:
            error_msg = f"Failed to end session: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
    def get_agent_desc(self) -> str:
        """Get agent description. Override for custom implementation."""
        self.logger.info(f"Getting agent description for {self.identifier}")
        return ""
