import re
from typing import Any, Dict, Optional

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import InvalidInputValueError


def get_config_or_default(config: Optional[Dict[str, Any]], config_key: str, default: Any):
    if config and config_key in config:
        return config[config_key]
    return default


def match_to_int(completion, regex_pattern, grade_choices):
    """
    Parse a completion string and return binary score based on grade.

    Args:
        completion: The completion string containing a grade
        regex_pattern: Regex pattern to extract the grade
        grade_choices: List of 2 strings [positive_grade, negative_grade]
                      e.g., ["C", "I"] for Complete/Incomplete
                      e.g., ["A", "N"] for Applicable/Not applicable

    Returns:
        1 if grade matches positive_grade, 0 if matches negative_grade
    """
    match = re.search(regex_pattern, completion)
    if not match:
        raise InvalidInputValueError(
            internal_code=ErrorCode.INVALID_JUDGE_RESPONSE_FORMAT_ERROR.value,
            message=f"Could not find the judge grade from the completion: {completion}",
        )
    grade = match.group(1).upper()  # Normalize to uppercase for comparison

    if grade == grade_choices[0].upper():
        correct_int = 1
    elif grade == grade_choices[1].upper():
        correct_int = 0
    else:
        raise InvalidInputValueError(
            internal_code=ErrorCode.INVALID_JUDGE_RESPONSE_FORMAT_ERROR.value,
            message=f"Invalid judge grade from the completion: {completion}",
        )
    return correct_int


def tally_votes(
    complete_cnt, incomplete_cnt, invalid_cnt, completions, regex_pattern, grade_choices
):
    for completion in completions:
        try:
            score = match_to_int(completion, regex_pattern, grade_choices)
            if score == 1:
                complete_cnt += 1
            elif score == 0:
                incomplete_cnt += 1
        except InvalidInputValueError:
            invalid_cnt += 1
    return complete_cnt, incomplete_cnt, invalid_cnt
