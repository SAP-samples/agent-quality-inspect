from typing import List

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.models.metrics.agent_trace import TurnTrace


class TraceValidator:
    """
    A utility class for validating agent turn traces.
    """

    @staticmethod
    def validate_turn_traces_content(agent_turn_traces: List[TurnTrace]) -> None:
        for turn_trace in agent_turn_traces:
            TraceValidator.validate_agent_input_if_empty(turn_trace)
            TraceValidator.validate_agent_response_if_empty(turn_trace)

    @staticmethod
    def validate_agent_trace_if_empty(agent_turn_traces: List[TurnTrace]) -> None:
        if not agent_turn_traces or len(agent_turn_traces) == 0:
            raise InvalidInputValueError(
                internal_code=ErrorCode.MISSING_VALUE.value,
                message="Agent turn traces list is empty or None.",
            )

    @staticmethod
    def validate_agent_input_if_empty(agent_turn_trace: TurnTrace) -> None:
        if not agent_turn_trace.agent_input:
            raise InvalidInputValueError(
                internal_code=ErrorCode.MISSING_VALUE.value,
                message=f"Turn :{agent_turn_trace.id} is missing agent input.",
            )

    @staticmethod
    def validate_agent_response_if_empty(agent_turn_trace: TurnTrace) -> None:
        if not agent_turn_trace.agent_response or not agent_turn_trace.agent_response.response:
            raise InvalidInputValueError(
                internal_code=ErrorCode.MISSING_VALUE.value,
                message=f"Turn :{agent_turn_trace.id} is missing agent response.",
            )
