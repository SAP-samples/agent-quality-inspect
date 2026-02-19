from dataclasses import dataclass
from typing import Optional


@dataclass
class TerminatingCondition:
    """
    Represents a condition used to terminate user-agent conversation early.
    """

    check: str
    """
    Description of the terminating condition.
    """
    stop_sequence: Optional[str] = None
    """
    The stop sequence indicating the end of the conversation when the condition is met. This is for future use and any value assigned here will be ignored for now.
    """
