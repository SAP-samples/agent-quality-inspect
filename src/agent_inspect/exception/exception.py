from agent_inspect.exception.error_codes import EvaluationComponent, UserProxyComponent, ToolComponent


class EvaluationError(Exception):
    """Base exception class for Evaluation errors."""

    def __init__(self, internal_code: str, message: str):
        self.internal_code = EvaluationComponent.EVALUATION_ERROR_CODE.value + internal_code
        self.message = f"Internal Code: {self.internal_code}, Error Message: {message}"
        super().__init__(self.message)

class InvalidInputValueError(ValueError):

    def __init__(self, internal_code: str, message: str, component_code=EvaluationComponent.EVALUATION_ERROR_CODE.value):
        self.internal_code = component_code + internal_code
        self.message = f"Internal Code: {self.internal_code}, Error Message: {message}"
        super().__init__(self.message)

class UserProxyError(Exception):

    def __init__(self, internal_code: str, message: str):
        self.internal_code = UserProxyComponent.USER_PROXY_ERROR_CODE.value + internal_code
        self.message = f"Internal Code: {self.internal_code}, Error Message: {message}"
        super().__init__(self.message)

class ToolError(Exception):
    """Base exception class for Tool errors."""

    def __init__(self, internal_code: str, message: str):
        self.internal_code = ToolComponent.TOOL_ERROR_CODE.value + internal_code
        self.message = f"Internal Code: {self.internal_code}, Error Message: {message}"
        super().__init__(self.message)
