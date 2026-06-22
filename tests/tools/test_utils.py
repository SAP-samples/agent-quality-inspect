import pytest

from agent_inspect.exception import EvaluationError, InvalidInputValueError
from agent_inspect.core.utils import match_to_int
from agent_inspect.tools.utils import (
    map_subgoal_validations_to_binary_matrix,
    validate_inputs_for_pass_k_initialisation,
)
from agent_inspect.metrics.constants import (
    COMPLETE_INCOMPLETE_GRADE_PATTERN,
    COMPLETE_INCOMPLETE_PAIR,
)


def test_match_to_int_returns_correct_int_for_valid_completion():
    completion = "Grade: C"
    result = match_to_int(completion, COMPLETE_INCOMPLETE_GRADE_PATTERN, COMPLETE_INCOMPLETE_PAIR)
    assert result == 1

    completion = "Grade: I"
    result = match_to_int(completion, COMPLETE_INCOMPLETE_GRADE_PATTERN, COMPLETE_INCOMPLETE_PAIR)
    assert result == 0


def test_match_to_int_raises_error_for_invalid_completion():
    completion = "This is a just dummy judge explanation.\n\nGrade: X"
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050003, Error Message: Could not find the judge grade from the completion: This is a just dummy judge explanation.\n\nGrade: X",
    ):
        match_to_int(completion, COMPLETE_INCOMPLETE_GRADE_PATTERN, COMPLETE_INCOMPLETE_PAIR)


def test_match_to_int_raises_error_when_no_match_found():
    completion = "No grade here"
    with pytest.raises(
        InvalidInputValueError,
        match="Internal Code: 050003, Error Message: Could not find the judge grade from the completion: No grade here",
    ):
        match_to_int(completion, COMPLETE_INCOMPLETE_GRADE_PATTERN, COMPLETE_INCOMPLETE_PAIR)


def test_map_subgoal_validations_handles_valid_completions():
    completions = ["Grade: C", "Grade: I", "Grade: C"]
    result = map_subgoal_validations_to_binary_matrix(completions)
    assert result == [1, 0, 1]


def test_map_subgoal_validations_skips_invalid_completions():
    completions = ["Grade: C", "Invalid Grade", "Grade: I"]
    result = map_subgoal_validations_to_binary_matrix(completions)
    assert result == [1, 0]


def test_map_subgoal_validations_returns_empty_for_all_invalid_completions():
    completions = ["Invalid Grade", "Another Invalid Grade"]
    result = map_subgoal_validations_to_binary_matrix(completions)
    assert result == []


def test_map_subgoal_validations_handles_empty_input():
    completions = []
    result = map_subgoal_validations_to_binary_matrix(completions)
    assert result == []


def test_validate_inputs_for_pass_k_raises_when_num_trials_is_none():
    with pytest.raises(EvaluationError, match="num_trials is invalid and must be provided"):
        validate_inputs_for_pass_k_initialisation(k_value=1, num_trials=None)


def test_validate_inputs_for_pass_k_raises_when_k_value_is_zero():
    with pytest.raises(EvaluationError, match="k_value \\(0\\) must be greater than 0"):
        validate_inputs_for_pass_k_initialisation(k_value=0, num_trials=5)


def test_validate_inputs_for_pass_k_raises_when_k_value_is_negative():
    with pytest.raises(EvaluationError, match="k_value \\(-1\\) must be greater than 0"):
        validate_inputs_for_pass_k_initialisation(k_value=-1, num_trials=5)


def test_validate_inputs_for_pass_k_raises_when_num_trials_is_zero():
    with pytest.raises(EvaluationError, match="num_trials is invalid and must be provided"):
        validate_inputs_for_pass_k_initialisation(k_value=1, num_trials=0)


def test_validate_inputs_for_pass_k_raises_when_num_trials_is_negative():
    with pytest.raises(EvaluationError, match="num_trials \\(-1\\) must be greater than 0"):
        validate_inputs_for_pass_k_initialisation(k_value=1, num_trials=-1)


def test_validate_inputs_for_pass_k_raises_when_k_greater_than_num_trials():
    with pytest.raises(
        EvaluationError,
        match="k_value \\(10\\) cannot be greater than num_trials \\(5\\)",
    ):
        validate_inputs_for_pass_k_initialisation(k_value=10, num_trials=5)


def test_validate_inputs_for_pass_k_succeeds_with_valid_inputs():
    # Should not raise any exception
    validate_inputs_for_pass_k_initialisation(k_value=3, num_trials=5)
    validate_inputs_for_pass_k_initialisation(k_value=1, num_trials=1)
