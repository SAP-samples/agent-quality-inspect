from typing import Optional, Any
from dataclasses import dataclass

@dataclass
class LLMPayload:
    """
    Represents the payload to be sent to a Language Model (LLM) for processing.
    """

    user_prompt: str
    """
    The raw text prompt provided by the user to the LLM for processing.
    """
    model: Optional[str] = None
    """
    The specific LLM model to be used for processing the prompt.
    """
    system_prompt: Optional[str] = None
    """
    The system-level prompt that provides context or instructions to the LLM.
    """
    temperature: Optional[float] = None
    """
    The temperature setting for the LLM, influencing the randomness of its output.
    """
    max_tokens: Optional[int] = None
    """
    The maximum number of tokens to be generated in the LLM's response.
    """
    structured_output: Optional[Any] = None
    """
    An optional structured format for the LLM's output, if applicable.
    """
