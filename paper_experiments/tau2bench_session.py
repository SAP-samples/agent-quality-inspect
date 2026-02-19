import sys
sys.path.append("../")
#TODO: Set the file paths correctly by setting in the __init__ folder and not using sys path
import time
from typing import Any, Dict
import os
import requests

from .session import BaseSession, SessionMessageResponse

class Tau2BenchSession(BaseSession):
    """Tau2Bench-specific session implementation."""

    def __init__(self, domain: str, agent_type: str = "azure/gpt-4.1"):
        """Initialize Tau2Bench session with domain-specific configuration."""
        self.domain = domain
        super().__init__(domain, agent_type)
    
    def get_config(self) -> Dict[str, Any]:
        """Return Tau2Bench configuration."""
        return {
            "base_url": os.environ.get("SERVER_ENDPOINT", "http://127.0.0.1:5500"),
            "endpoints": {
                "start": "/start_conversation",
                "message": "/conversation/{session_id}/message",
                "trajectory": "/conversation/{session_id}/trajectory",
                "end": "/conversation/{session_id}/end"
            }
        }
    
    def send_message(self, message: str, retries: int = 3) -> SessionMessageResponse:
        """Send a message to the current session."""
        self._ensure_session_active()
        
        url = f"{self.base_url}{self.endpoints['message'].format(session_id=self.session_id)}"
        payload = self.build_message_payload(message)
        
        self.logger.info(f"Sending message to session {self.session_id}: {message}")
        for attempt in range(retries):
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return SessionMessageResponse(response_message=response.json().get("response", ""))
            else:
                self.logger.error(f"Attempt {attempt + 1} failed: {response.status_code} - {response.text}")
                self.logger.info("Retrying...")
                time.sleep(3)

        error_msg = f"Failed to send message after {retries} attempts."
        self.logger.error(error_msg)
        raise Exception(error_msg)
    
    def build_start_payload(self) -> Dict[str, Any]:
        """Create payload for starting a Tau2Bench session."""
        return {
            "agent_llm": self.agent_type,
            "domain": self.domain,
        }
    
    def process_trajectory(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Tau2Bench trajectory response data."""
        return response_data["trajectory"]
    
    def get_agent_desc(self) -> str:
        """Get agent description for Tau2Bench domain."""
        url = f"{self.base_url}/agent_description"
        self.logger.info(f"Retrieving agent description for domain {self.domain}")
        payload = {"domain": self.domain}
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json().get("description", "")
        else:
            error_msg = f"Failed to retrieve agent description: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
