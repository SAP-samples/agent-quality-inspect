import logging
from typing import List

from agent_inspect.exception.error_codes import UserProxyComponent, ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.models.user_proxy.terminating_condition import (
    TerminatingCondition,
    TaskCompletedTerminatingCondition,
    TaskDelegatedTerminatingCondition,
    TaskBlockedTerminatingCondition,
)


logger = logging.getLogger(__name__)


class UserProxyInputValidator:
    @staticmethod
    def validate_task_summary(task_summary: str):
        if not task_summary or task_summary.strip() == "":
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.MISSING_VALUE.value,
                message="Task summary cannot be empty to create User Proxy.",
            )

    @staticmethod
    def replace_empty_strings_fields_with_defaults(
        terminating_condition: TerminatingCondition,
    ):
        if terminating_condition.check.strip() == "":
            terminating_condition.check = terminating_condition.__class__.__dataclass_fields__[
                "check"
            ].default
            logger.warning(
                "Terminating condition check was given as empty string. Replaced with default value."
            )
        if terminating_condition.stop_sequence.strip() == "":
            terminating_condition.stop_sequence = (
                terminating_condition.__class__.__dataclass_fields__["stop_sequence"].default
            )
            logger.warning(
                "Terminating condition stop_sequence was given as empty string. Replaced with default value."
            )

    @staticmethod
    def validate_single_terminating_condition(
        terminating_conditions: List[TerminatingCondition],
    ):
        if len(terminating_conditions) != 1:
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.INVALID_VALUE.value,
                message="Exactly one terminating condition must be provided when multi terminating conditions flag is not set.",
            )
        if type(terminating_conditions[0]) is not TerminatingCondition:
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.INVALID_VALUE.value,
                message="Terminating condition must be of type TerminatingCondition and not its subclasses when multi terminating conditions flag is not set.",
            )
        UserProxyInputValidator.replace_empty_strings_fields_with_defaults(
            terminating_conditions[0]
        )

    @staticmethod
    def validate_multiple_terminating_conditions(
        terminating_conditions: List[TerminatingCondition],
    ):
        completed_condition_count = sum(
            1
            for condition in terminating_conditions
            if isinstance(condition, TaskCompletedTerminatingCondition)
        )
        delegated_condition_count = sum(
            1
            for condition in terminating_conditions
            if isinstance(condition, TaskDelegatedTerminatingCondition)
        )
        blocked_condition_count = sum(
            1
            for condition in terminating_conditions
            if isinstance(condition, TaskBlockedTerminatingCondition)
        )
        base_condition_count = sum(
            1 for condition in terminating_conditions if type(condition) is TerminatingCondition
        )

        if base_condition_count > 0:
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.INVALID_VALUE.value,
                message="Terminating conditions must be of type TaskCompletedTerminatingCondition, "
                "TaskDelegatedTerminatingCondition, or TaskBlockedTerminatingCondition when multi terminating conditions flag is set.",
            )

        if completed_condition_count > 1:
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.INVALID_VALUE.value,
                message="Only one task completed terminating condition can be provided.",
            )

        if delegated_condition_count > 1:
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.INVALID_VALUE.value,
                message="Only one task delegated terminating condition can be provided.",
            )

        if blocked_condition_count > 1:
            raise InvalidInputValueError(
                component_code=UserProxyComponent.USER_PROXY_ERROR_CODE.value,
                internal_code=ErrorCode.INVALID_VALUE.value,
                message="Only one task blocked terminating condition can be provided.",
            )

        for terminating_condition in terminating_conditions:
            UserProxyInputValidator.replace_empty_strings_fields_with_defaults(
                terminating_condition
            )

    @staticmethod
    def validate_terminating_conditions(
        terminating_conditions: List[TerminatingCondition], allow_multiple: bool
    ):
        if allow_multiple:
            UserProxyInputValidator.validate_multiple_terminating_conditions(terminating_conditions)
        else:
            UserProxyInputValidator.validate_single_terminating_condition(terminating_conditions)
