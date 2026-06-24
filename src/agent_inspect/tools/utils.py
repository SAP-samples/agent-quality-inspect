from typing import List

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import EvaluationError, InvalidInputValueError
from agent_inspect.core.utils import match_to_int
from agent_inspect.metrics.constants import (
    COMPLETE_INCOMPLETE_GRADE_PATTERN,
    COMPLETE_INCOMPLETE_PAIR,
)


def map_subgoal_validations_to_binary_matrix(completions: List[str]) -> List[int]:
    binary_matrix = []
    for completion in completions:
        try:
            # Supports both C/I (completion) and A/N (applicability) grades
            score = match_to_int(
                completion, COMPLETE_INCOMPLETE_GRADE_PATTERN, COMPLETE_INCOMPLETE_PAIR
            )
            binary_matrix.append(score)
        except InvalidInputValueError:
            # TODO: assume the completion includes the specific matching pattern
            continue  # Skip invalid responses
    return binary_matrix


def validate_inputs_for_pass_k_initialisation(k_value: int, num_trials: int):

    if not num_trials:
        raise EvaluationError(
            ErrorCode.INVALID_VALUE.value, "num_trials is invalid and must be provided."
        )

    if k_value <= 0:
        raise EvaluationError(
            ErrorCode.INVALID_VALUE.value, f"k_value ({k_value}) must be greater than 0"
        )

    if num_trials <= 0:
        raise EvaluationError(
            ErrorCode.INVALID_VALUE.value,
            f"num_trials ({num_trials}) must be greater than 0",
        )

    if k_value > num_trials:
        raise EvaluationError(
            ErrorCode.INVALID_VALUE.value,
            f"k_value ({k_value}) cannot be greater than num_trials ({num_trials})",
        )
