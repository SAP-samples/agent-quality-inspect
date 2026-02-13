from typing import List

from agent_inspect.exception.error_codes import UserProxyComponent, ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.models.user_proxy.terminating_condition import TerminatingCondition



class UserProxyInputValidator:
    @staticmethod
    def validate_terminating_condition(terminating_conditions: List[TerminatingCondition]):
        if not terminating_conditions:
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.MISSING_VALUE.value,
                message="At least one terminating condition must be provided to create User Proxy."
            )

        for terminating_condition in terminating_conditions:
            if not terminating_condition.check.strip():
                raise InvalidInputValueError(
                    component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                    internal_code=ErrorCode.MISSING_VALUE.value,
                    message="Terminating check cannot be an empty string."
                )

    @staticmethod
    def validate_task_summary(task_summary: str):
        if not task_summary or task_summary.strip() == "":
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.MISSING_VALUE.value,
                message="Task summary cannot be empty to create User Proxy."
            )
