from typing import Optional, List
from dataclasses import dataclass

from agent_inspect.models.metrics.agent_data_sample import SubGoal, ExpectedToolCall


@dataclass
class ValidationResult:
    """
    Represents a result produced after validation (e.g. subgoal validation, tool call validation).
    """

    is_completed: bool
    """
    Indicates whether the validation was completed successfully.
    """
    explanations: List[str]
    """
    Contains a list of explanation/reason(s) for why/how this particular validation result is produced.
    """

@dataclass
class SubGoalValidationResult(ValidationResult):
    """
    Represents a result produced after validation.
    """
    sub_goal: SubGoal
    """
    Contains the subgoal that is being validated.
    """
    prompt_sent_to_llmj: Optional[str] = None
    """
    The entire prompt that is sent to the LLM-as-a-judge for validation.
    """


@dataclass
class ToolCallValidationResult(ValidationResult):
    """
    Represents a tool correctness result after validation.
    """
    
    expected_tool_call: ExpectedToolCall
    """
    The expected ground truth tool call that is being validated against.
    """

