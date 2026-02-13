from dataclasses import dataclass
from typing import Optional, List, Union, Any


@dataclass
class SubGoal:
    """
    A subgoal is a natural language assertion that defines the success criteria of an agent within a larger task. Subgoals specify intermediate success criteria and can also include the final goal.
    eg:
    "Agent should call search_messages after getting the current timestamp"
    "Agent should inform the user: Your oldest message says 'Hey kid, you want some GPU?'.
    """

    details: str
    """
    Details containing the information and criteria used by the metric to determine when the agent has achieved this subgoal.
    """
    type: Optional[str] = None
    """
    Represents the type of subgoal (e.g. grading notes).
    """
    turn: Optional[Union[int|str]] = None
    """
    Represents which turn(s) should this subgoal be considered. Optional and this is only a placeholder for future implementation.
    """


@dataclass
class ToolInputParameter:
    """
    Represents a parameter which the agent should invoke with the tool.
    """

    name: str
    """
    Represents the expected name of the parameter (variable name).
    """
    value: Optional[Any] = None
    """
    Represents the expected value of the parameter at the moment the tool is invoked. This will be converted to str during metric calculation.
    """
    check: Optional[str] = None
    """
    Represents the llm prompt to check the correctness of the parameter name and the value that the agent invokes with the tool.
    """

@dataclass
class ToolOutput:
    """
    Represents the expected output from the tool after the agent invokes it.
    """

    value: Optional[Any] = None
    """
    Represents the expected output value from the tool after the agent invokes it. This will be converted to str during metric calculation.
    """
    check: Optional[str] = None
    """
    Represents the llm prompt to check the correctness of the output value from the tool after the agent invokes it.
    """


@dataclass
class ExpectedToolCall:
    """
    Represents the correct tool invocation an agent is expected to make for a particular task. It serves as the ground-truth tool call for the evaluations.
    """

    tool: str
    """
    Represents a tool that should be called or utilized by the agent at the time of evaluation. Can be a name, a description of the tool, the url of api call, etc.
    """
    expected_parameters: Optional[List[ToolInputParameter]] = None
    """
    A list of parameters with which the tool should be called by the agent during the time of evaluation.
    """
    expected_output: Optional[ToolOutput] = None
    """
    Represents the expected output from the tool after the agent invokes it.
    """
    turn: Optional[int | str] = None
    """
    Represents which turn(s) should this tool call be considered. Optional and this is only a placeholder for future implementation.
    """


@dataclass
class Conversation:
    """
    Represents one back-and-forth exchange between a user and an agent (one turn), containing an input message and an optional expected response.
    """

    turn_id: int
    """
    ID representing the sequence number of conversation in the list of conversations.
    """
    message: str
    """
    Input message to the agent.
    """
    expected_response: Optional[str] = None
    """
    Expected response from the agent given the agent input message. This can be none during the evaluation with user proxy.
    """


@dataclass
class EvaluationSample:
    """
    Represents an item in the evaluation dataset.
    """

    sub_goals: List[SubGoal]
    """
    A list of sub goals which should be achieved by an agent during an evaluation run.
    """
    id: Optional[int] = None
    """
    Unique identifier for the evaluation sample.
    """
    expected_tool_calls: Optional[List[ExpectedToolCall]] = None
    """
    A list of expected tools that an agent should call or utilize during an evaluation run. (e.g. API calls, calculator, web search, etc.)
    """
    conversation: Optional[List[Conversation]] = None
    """
    A list of conversation between an agent and a user/user proxy. The sequence of conversations in this list, matters for metric to understand which comes after next.
    """
    user_instruction: Optional[str] = None
    """
    An instruction/instructions for user proxy how it should behave/response while communicating with the agent during an evaluation run.
    """

