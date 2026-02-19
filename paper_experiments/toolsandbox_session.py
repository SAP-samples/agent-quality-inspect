import sys
sys.path.append("../")
#TODO: Set the file paths correctly by setting in the __init__ folder and not using sys path

from typing import Any, Dict
import os
import requests
import time

from .session import BaseSession, SessionMessageResponse

class ToolsandboxSession(BaseSession):
    """Tool sandbox session implementation."""
    
    def __init__(self, scenario_name: str, agent_type: str = "GPT_4_1"):
        """Initialize ToolSandbox session with scenario-specific configuration."""
        self.scenario_name = scenario_name
        super().__init__(scenario_name, agent_type)
    
    def get_config(self) -> Dict[str, Any]:
        """Return ToolSandbox configuration."""
        return {
            "base_url": os.environ.get("SERVER_ENDPOINT", "http://localhost:8000"),
            "endpoints": {
                "start": "/start_session",
                "message": "/message",
                "trajectory": "/trajectory/{session_id}",
                "end": "/end_session/{session_id}"
            }
        }
    
    def send_message(self, message: str, retries: int = 3) -> Dict[str, Any]:
        """Send a message to the current session."""
        self._ensure_session_active()
        
        url = f"{self.base_url}{self.endpoints['message'].format(session_id=self.session_id)}"
        payload = self.build_message_payload(message)
        
        self.logger.info(f"Sending message to session {self.session_id}: {message}")
        for attempt in range(retries):
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                content = response.json()['content']
                return SessionMessageResponse(response_message=content)
            else:
                self.logger.error(f"Attempt {attempt + 1} failed: {response.status_code} - {response.text}")
                self.logger.info("Retrying...")
                time.sleep(3)

        error_msg = f"Failed to send message after {retries} attempts."
        self.logger.error(error_msg)
        raise Exception(error_msg)
    
    def build_start_payload(self) -> Dict[str, Any]:
        """Create payload for starting a ToolSandbox session."""
        return {
            "scenario": self.scenario_name,
            "agent_type": self.agent_type,
        }
    
    def process_trajectory(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ToolSandbox trajectory response data."""
        return response_data
    
    def build_message_payload(self, message: str) -> Dict[str, Any]:
        """Create payload for sending a message in ToolSandbox."""
        return {
            "scenario": self.scenario_name,
            "session_id": self.session_id,
            "message": message,
            "agent_type": self.agent_type,
        }
    

    
    def get_agent_desc(self) -> str:
        """Get agent description for ToolSandbox scenario."""
        self.logger.info(f"Retrieving agent description for scenario {self.scenario_name}")
        
        # Add domain-specific logic here as needed
        # Example:
        # if self.scenario_name == "search_message_with_recency_latest_multiple_user_turn":
        #     return "Search Message Agent with access to get_timestamp function..."
        
        return ""

        
    def end(self) -> None:
        self.session_id = None