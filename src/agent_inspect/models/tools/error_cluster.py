"""Error cluster definitions for error analysis.

This module defines error cluster types used by error analysis implementations.
Error clusters categorize common failure modes in agent behavior.
"""

from dataclasses import dataclass
from abc import ABC
from typing import List


@dataclass(frozen=True)
class ErrorCluster(ABC):
    """Base class for error cluster definitions.

    Error clusters represent categories of agent failures. Each cluster has:
    - cluster_label: A short, human-readable name for the error type
    - description: A detailed explanation of when this error occurs

    This dataclass is frozen (immutable) to prevent accidental modification
    of the default cluster definitions.

    Attributes:
        cluster_label: Short name identifying the error type
        description: Detailed explanation of the error category
    """

    cluster_label: str
    description: str

    def to_dict(self) -> dict:
        """Convert cluster to dictionary format for LLM prompts.

        Returns:
            Dictionary with cluster_label and description keys
        """
        return {"cluster_label": self.cluster_label, "description": self.description}


# These dataclasses are frozen to ensure immutability of the default cluster definitions


@dataclass(frozen=True)
class IncorrectToolInputCluster(ErrorCluster):
    """Agent called a tool with incorrect, missing, or malformed input parameters."""

    cluster_label: str = "Incorrect Tool Input"
    description: str = "Agent called a tool with incorrect, missing, or malformed input parameters."


@dataclass(frozen=True)
class IncorrectToolOutputHandlingCluster(ErrorCluster):
    """Agent failed to correctly interpret, parse, or use the output returned by a tool."""

    cluster_label: str = "Incorrect Tool Output Handling"
    description: str = (
        "Agent failed to correctly interpret, parse, or use the output returned by a tool."
    )


@dataclass(frozen=True)
class WrongToolSelectionCluster(ErrorCluster):
    """Agent selected the wrong tool for the task or called tools in an incorrect sequence."""

    cluster_label: str = "Wrong Tool Selection"
    description: str = (
        "Agent selected the wrong tool for the task or called tools in an incorrect sequence."
    )


@dataclass(frozen=True)
class MissedToolCallCluster(ErrorCluster):
    """Agent failed to call a necessary tool to complete the subgoal."""

    cluster_label: str = "Missed Tool Call"
    description: str = "Agent failed to call a necessary tool to complete the subgoal."


@dataclass(frozen=True)
class InstructionFollowingErrorCluster(ErrorCluster):
    """Agent failed to follow explicit instructions or constraints from the user."""

    cluster_label: str = "Instruction Following Error"
    description: str = "Agent failed to follow explicit instructions or constraints from the user."


@dataclass(frozen=True)
class IncompleteCommunicationCluster(ErrorCluster):
    """Agent's response is missing necessary information or is too vague to be useful."""

    cluster_label: str = "Incomplete or Vague Communication"
    description: str = (
        "Agent's response is missing necessary information or is too vague to be useful."
    )


@dataclass(frozen=True)
class FaithfulnessErrorCluster(ErrorCluster):
    """Agent's response contradicts or is not supported by the output of the tools it used."""

    cluster_label: str = "Faithfulness Error (Ungrounded)"
    description: str = (
        "Agent's response contradicts or is not supported by the output of the tools it used."
    )


@dataclass(frozen=True)
class LogicalReasoningErrorCluster(ErrorCluster):
    """Agent's reasoning process was flawed, got stuck in a loop, or gave up before the task was complete."""

    cluster_label: str = "Logical Reasoning Error"
    description: str = "Agent's reasoning process was flawed, got stuck in a loop, or gave up before the task was complete."


@dataclass(frozen=True)
class UnclassifiedErrorCluster(ErrorCluster):
    """Errors that do not fit into any of the other predefined categories."""

    cluster_label: str = "Unclassified"
    description: str = "Errors that do not fit into any of the other predefined error categories. "
    "Also use this when you cannot identify a clear agent behavior or decision that caused the failure."


# Default cluster lists

DEFAULT_SUBGOAL_ERROR_CLUSTERS: List[ErrorCluster] = [
    IncorrectToolInputCluster(),
    IncorrectToolOutputHandlingCluster(),
    WrongToolSelectionCluster(),
    MissedToolCallCluster(),
    InstructionFollowingErrorCluster(),
    IncompleteCommunicationCluster(),
    FaithfulnessErrorCluster(),
    LogicalReasoningErrorCluster(),
    UnclassifiedErrorCluster(),
]

DEFAULT_TOOL_CALL_ERROR_CLUSTERS: List[ErrorCluster] = [
    WrongToolSelectionCluster(),
    IncorrectToolInputCluster(),
    IncorrectToolOutputHandlingCluster(),
    UnclassifiedErrorCluster(),
]
