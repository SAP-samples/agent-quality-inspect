from dataclasses import dataclass
from typing import Optional, List, Union, Any
from agent_inspect.models.metrics.agent_data_sample import ToolInputParameter


@dataclass
class AgentResponse:
    """
    Represents an agent response produced by the agent in one turn.
    """

    response: Union[str|dict]
    """
    A response from the agent, which can either be a python str or a python dict. This will be converted to a str during metric calculation if a dict is provided.
    """
    status_code: Optional[str] = None
    """
    Contains the http status code from agent if any.
    """


@dataclass
class Step:
    """
    Represents a necessary step that the agent takes within a turn to produce the final response(s) back to the user/ user proxy.
    """

    id: str
    """
    Represents an ID assigned to a step, must be a unique identifier.
    """
    parent_ids: List[str]
    """
    Contains a list of parent step ID(s) which the agent took before the current step (e.g. ID(s) of the immediate previous steps.)
    """
    tool: Optional[str] = None
    """
    Represents a tool that was called or utilized by the agent during the current step. Can be a name, a description of the tool, the url of api call, etc. Optional if no tool is called in current step.
    """
    tool_input_args: Optional[List[ToolInputParameter]] = None
    """
    A list of parameters which the tool was called by the agent during the time of evaluation.
    """
    tool_output: Optional[Any] = None
    """
    Contains the tool output produced by the agent in the current step. Optional and can be none if no output was produced.
    """
    agent_thought: Optional[str] = None
    """
    Represent the agent's thinking process (what the agent is thinking) at the current step.
    """
    input_token_consumption: Optional[int] = None
    """
    Represents the total number of input tokens consumed in the current step.
    """
    output_token_consumption: Optional[int] = None
    """
    Represents the total number of output tokens consumed in the current step.
    """
    reasoning_token_consumption: Optional[int] = None
    """
    Represents the total number of tokens consumed for reasoning purposes in the current step.
    """


@dataclass
class TurnTrace:
    """
    Represents a turn: one back and forth conversation between the agent and the user/user proxy by encapsulating the agent trace (logs) produced during a turn.
    """
    
    id: str
    """
    Represents an ID assigned to a turn, must be a unique identifier.
    """
    agent_input: str
    """
    An input string to the agent from a user/user proxy.
    """
    agent_response: Optional[AgentResponse] = None
    """
    An agent response produced by the agent given the input message (agent_input).
    """
    from_id: Optional[str] = None
    """
    Represents an ID of the parent turn that produces the current step. Optional and this is only a placeholder for future implementation.
    """
    steps: Optional[List[Step]] = None
    """
    A list of steps (e.g. tool calls, thinking processes, etc.) that the agent takes to finally produce agent response(s).
    """
    latency_in_ms: Optional[float] = None
    """
    Represent the total time taken by the agent to produce the output response(s).
    """


@dataclass
class AgentDialogueTrace:
    """
    Represents an agent trace (logs) produced by an agent given an evaluation data sample during an evaluation run.
    """
    
    turns: Optional[List[TurnTrace]] = None
    """
    A list of turn which contains the subset of agent trace (logs) produced during one back and forth conversation between the agent and the user/user proxy.
    """
