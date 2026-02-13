from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.models.metrics.agent_data_sample import ExpectedToolCall


class ExpectedToolCallValidator:
    """
    A utility class for validating expected tool call in eval data samples.
    """
    
    @staticmethod
    def validate_expected_tool_call(expected_tool_call: ExpectedToolCall) -> None:
        if not expected_tool_call.tool:
            raise InvalidInputValueError(internal_code=ErrorCode.MISSING_VALUE.value, message="ExpectedToolCall is missing Tool Name.")
