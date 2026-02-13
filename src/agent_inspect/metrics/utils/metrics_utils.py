import re
from typing import Any, Dict, List

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import EvaluationError
from agent_inspect.metrics.constants import DEFAULT_GRADE_PATTERN, STATUS_200


def get_majority_voted_score(score_to_vote_count: Dict[Any, int]):
    return max(score_to_vote_count, key=score_to_vote_count.get)

def get_config_or_default(config: Dict[str, Any], config_key: str, default: Any):
    if config and config_key in config:
        return config[config_key]
    return default

def match_to_int(completion):
    match = re.search(DEFAULT_GRADE_PATTERN, completion)
    if not match:
        raise EvaluationError(internal_code=ErrorCode.INVALID_JUDGE_RESPONSE_FORMAT_ERROR.value,
                              message=f"Could not find the judge grade from the completion: {completion}")
    if match.group(1) == "C":
        correct_int = 1
    elif match.group(1) == "I":
        correct_int = 0
    else:
        raise EvaluationError(internal_code=ErrorCode.INVALID_JUDGE_RESPONSE_FORMAT_ERROR.value,
                              message=f"Invalid judge grade from the completion: {completion}")
    return correct_int

def map_subgoal_validations_to_binary_matrix(completions: List[str]) -> List[int]:
    binary_matrix = []
    for completion in completions:
        try:
            score = match_to_int(completion)
            binary_matrix.append(score)
        except EvaluationError:
            # TODO: assume the completion includes the specific matching pattern
            continue  # Skip invalid responses
    return binary_matrix

def tally_votes(complete_cnt, incomplete_cnt, invalid_cnt, completions):
    for completion in completions:
        try:
            score = match_to_int(completion)
            if score == 1:
                complete_cnt += 1
            elif score == 0:
                incomplete_cnt += 1
        except EvaluationError:
            invalid_cnt += 1
    return complete_cnt, incomplete_cnt, invalid_cnt

def tally_judge_voting(complete_cnt, incomplete_cnt, invalid_cnt, judge_responses):
    completions = []
    for judge_response in judge_responses:
        if judge_response.status != STATUS_200:
            invalid_cnt += 1
        else:
            completions.append(judge_response.completion)
    complete_cnt, incomplete_cnt, invalid_cnt = tally_votes(complete_cnt, incomplete_cnt, invalid_cnt, completions)
    return complete_cnt, incomplete_cnt, invalid_cnt

def validate_inputs_for_pass_k_initialisation(k_value: int, num_trials: int):
        
    if not num_trials:
        raise EvaluationError(ErrorCode.INVALID_VALUE.value, "num_trials is invalid and must be provided.")
    
    if k_value <= 0:
            raise EvaluationError(ErrorCode.INVALID_VALUE.value, f"k_value ({k_value}) must be greater than 0")
        
    if num_trials <= 0:
        raise EvaluationError(ErrorCode.INVALID_VALUE.value, f"num_trials ({num_trials}) must be greater than 0")
    
    if k_value > num_trials:
        raise EvaluationError(ErrorCode.INVALID_VALUE.value, f"k_value ({k_value}) cannot be greater than num_trials ({num_trials})")