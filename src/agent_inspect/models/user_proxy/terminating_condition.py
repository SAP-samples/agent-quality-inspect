from dataclasses import dataclass


DEFAULT_STOP_SEQUENCE = "END_CONVERSATION"
DEFAULT_DONE_STOP_SEQUENCE = "##DONE##"
DEFAULT_DELEGATED_STOP_SEQUENCE = "##DELEGATE##"
DEFAULT_BLOCKED_STOP_SEQUENCE = "##BLOCKED##"

DEFAULT_CHECK = "The task is considered complete if the instruction goal is satisfied or you are transferred to another agent or you find yourself in a situation in which the scenario does not provide enough information for you to continue the conversation."
DEFAULT_DONE_CHECK = "The task is considered complete when the instruction goal as defined in the [user_task_summary] has been satisfied, even if you (the LLM-simulated user) do not have sufficient information to independently verify the correctness of the outcome."
DEFAULT_DELEGATED_CHECK = "The task must be terminated as soon as the AI agent initiates, implies, or begins any action that transfers, is transferring, or is in the process of transferring—explicitly or implicitly—control of the conversation to another agent, system, or human operator, even if the instruction goal as defined in the [user_task_summary] has not been fully satisfied."
DEFAULT_BLOCKED_CHECK = "The task must be terminated when the AI agent is unable to proceed because it lacks the required permissions, knowledge, capabilities, or access to the necessary tools to fulfill the instruction goal as defined in the [user_task_summary]."


@dataclass
class TerminatingCondition:
    """
    Represents a condition used to terminate user-agent conversation early.
    """

    check: str = DEFAULT_CHECK
    """
    Description of the terminating condition.
    """
    stop_sequence: str = DEFAULT_STOP_SEQUENCE
    """
    The stop sequence indicating the end of the conversation when the condition is met. This is for future use and any value assigned here will be ignored for now.
    """


@dataclass
class TaskCompletedTerminatingCondition(TerminatingCondition):
    """
    Represents a terminating condition indicating that the task is completed.
    """

    check: str = DEFAULT_DONE_CHECK
    """
    The default check for task completion.
    """
    stop_sequence: str = DEFAULT_DONE_STOP_SEQUENCE
    """
    The default stop sequence for completed task.
    """


@dataclass
class TaskDelegatedTerminatingCondition(TerminatingCondition):
    """
    Represents a terminating condition indicating that the task is delegated to another party.
    """

    check: str = DEFAULT_DELEGATED_CHECK
    """
    The default check for task delegation.
    """
    stop_sequence: str = DEFAULT_DELEGATED_STOP_SEQUENCE
    """
    The default stop sequence for delegated task.
    """


@dataclass
class TaskBlockedTerminatingCondition(TerminatingCondition):
    """
    Represents a terminating condition indicating that the task is blocked due to some issues.
    """

    check: str = DEFAULT_BLOCKED_CHECK
    """
    The default check for task being blocked.
    """
    stop_sequence: str = DEFAULT_BLOCKED_STOP_SEQUENCE
    """
    The default stop sequence for blocked task.
    """
