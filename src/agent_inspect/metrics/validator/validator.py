from abc import abstractmethod
from typing import Optional, Dict, Any, List

from agent_inspect.metrics.utils.metrics_utils import tally_votes, get_config_or_default

from agent_inspect.metrics.constants import STATUS_200, OPTIMIZE_JUDGE_TRIALS, MAX_RETRY_JUDGE_TRIALS, \
    MAX_RETRY_JUDGE_TRIALS_DEFAULT, NUM_JUDGE_TRIALS, NUM_JUDGE_TRIALS_DEFAULT, COULD_NOT_REACH_MAJORITY_DECISION

from agent_inspect.exception.error_codes import ErrorCode

from agent_inspect.exception import EvaluationError

from agent_inspect.models.metrics.agent_trace import TurnTrace

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.models.llm_response import LLMResponse
from agent_inspect.models.metrics.validation_result import ValidationResult


class Validator:
    """
    Abstract class which should be extended for actual implementation of validators.

    :param llm_client: the client which allows connection to the llm-as-a-judge model for evaluations.
    :param config: configuration for validator initialization. Default to ``None``.
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm_client = llm_client

    @abstractmethod 
    async def validate(self, turn_traces: List[TurnTrace], kwargs) -> ValidationResult:
        """
        This is an abstract method and should be implemented in a concrete class.

        :param turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :param kwargs: Additional keyword arguments that may be required for specific validation logic. These arguments can be used to pass optional parameters or configuration settings to the validator.
        :return: a :obj:`~agent_inspect.models.metrics.validation_result.ValidationResult` object containing the validation output.
        """
        ...

    async def get_majority_voted_score_from_judge_responses(self, prompt: str) -> (
            int, List[str]):
        optimize_judge_trials = get_config_or_default(config=self.config, config_key=OPTIMIZE_JUDGE_TRIALS,
                                                      default=False)
        max_retry_judge_trials = get_config_or_default(config=self.config, config_key=MAX_RETRY_JUDGE_TRIALS,
                                                       default=MAX_RETRY_JUDGE_TRIALS_DEFAULT)
        num_judge_trials = get_config_or_default(config=self.config, config_key=NUM_JUDGE_TRIALS,
                                                 default=NUM_JUDGE_TRIALS_DEFAULT)
        if optimize_judge_trials:
            return await Validator.get_majority_voted_score_from_judge_responses_optimised(
                self.llm_client, prompt, num_judge_trials)
        else:
            return await Validator.get_majority_voted_score_from_judge_responses_unoptimised(
                self.llm_client, prompt, num_judge_trials, max_retry_judge_trials)


    @staticmethod
    async def get_majority_voted_score_from_judge_responses_unoptimised(llm_client: LLMClient, prompt: str, no_of_trials: int, max_retry_judge_trials: int) -> (
            int, List[str]):
        Validator._validate_judge_trials(no_of_trials)
        judge_explanations = []
        prompts = [prompt] * no_of_trials
        judge_responses = await llm_client.make_llm_requests(prompts)
        judge_explanations.extend(Validator._get_judge_explanations_from_responses(judge_responses))
        completed_trial_count = incompleted_trial_cnt = invalid_trial_count = 0
        completed_trial_count, incompleted_trial_cnt, invalid_trial_count = Validator._tally_judge_voting(
            completed_trial_count, incompleted_trial_cnt, invalid_trial_count, judge_responses)
        # Retry logic for invalid trials
        retry_attempts = 0
        while invalid_trial_count > 0 and retry_attempts < max_retry_judge_trials:
            retry_prompts = [prompt] * invalid_trial_count
            retry_responses = await llm_client.make_llm_requests(retry_prompts)
            judge_explanations.extend(
                Validator._get_judge_explanations_from_responses(retry_responses))
            new_completed, new_incompleted, new_invalid = Validator._tally_judge_voting(
                0, 0, 0, retry_responses)
            completed_trial_count += new_completed
            incompleted_trial_cnt += new_incompleted
            invalid_trial_count = new_invalid
            retry_attempts += 1
        if invalid_trial_count > 0:
            raise EvaluationError(internal_code=ErrorCode.INVALID_LLM_JUDGE_RESULT_ERROR.value,
                                  message="One or more judge trials returned invalid responses after retries.")
        if completed_trial_count > incompleted_trial_cnt:
            return 1, judge_explanations
        else:
            return 0, judge_explanations


    @staticmethod
    async def get_majority_voted_score_from_judge_responses_optimised(llm_client: LLMClient, prompt: str,
                                                                      no_of_trials: int) -> (
            int, List[str]):
        Validator._validate_judge_trials(no_of_trials)
        judge_explanations = []
        threshold = (no_of_trials // 2) + 1
        completed_trial_count = incompleted_trial_cnt = invalid_trial_count = 0
        processed = 0
        first_wave = min(threshold, no_of_trials)
        prompts = [prompt] * first_wave
        judge_responses = await llm_client.make_llm_requests(prompts)
        judge_explanations.extend(Validator._get_judge_explanations_from_responses(judge_responses))
        completed_trial_count, incompleted_trial_cnt, invalid_trial_count = Validator._tally_judge_voting(
            completed_trial_count, incompleted_trial_cnt, invalid_trial_count, judge_responses)
        processed += first_wave
        if completed_trial_count >= threshold:
            return 1, judge_explanations
        if incompleted_trial_cnt >= threshold:
            return 0, judge_explanations

        while processed < no_of_trials:
            remaining = no_of_trials - processed
            if completed_trial_count + remaining < threshold and incompleted_trial_cnt + remaining < threshold:
                raise EvaluationError(internal_code=ErrorCode.INSUFFICIENT_JUDGE_RESPONSES_ERROR.value,
                                      message=COULD_NOT_REACH_MAJORITY_DECISION)
            required_completion_count = max(0, threshold - completed_trial_count)
            required_incomplete_count = max(0, threshold - incompleted_trial_cnt)
            wave = min(remaining, min(required_completion_count, required_incomplete_count))
            prompts = [prompt] * wave
            judge_responses = await llm_client.make_llm_requests(prompts)
            judge_explanations.extend(
                Validator._get_judge_explanations_from_responses(judge_responses))
            completed_trial_count, incompleted_trial_cnt, invalid_trial_count = Validator._tally_judge_voting(
                completed_trial_count, incompleted_trial_cnt, invalid_trial_count,
                judge_responses)
            processed += wave
            if completed_trial_count >= threshold:
                return 1, judge_explanations
            if incompleted_trial_cnt >= threshold:
                return 0, judge_explanations
        raise EvaluationError(internal_code=ErrorCode.INSUFFICIENT_JUDGE_RESPONSES_ERROR.value,
                              message=COULD_NOT_REACH_MAJORITY_DECISION)
    
    @staticmethod
    def _tally_judge_voting(complete_cnt, incomplete_cnt, invalid_cnt, judge_responses):
        completions = []
        for judge_response in judge_responses:
            if judge_response.status != STATUS_200 or not judge_response.completion or not judge_response.completion.strip():
                invalid_cnt += 1
            else:
                completions.append(judge_response.completion)
        complete_cnt, incomplete_cnt, invalid_cnt = tally_votes(complete_cnt, incomplete_cnt, invalid_cnt, completions)
        return complete_cnt, incomplete_cnt, invalid_cnt

    @staticmethod
    def _get_judge_explanations_from_responses(judge_responses: List[LLMResponse]) -> List[str]:
        explanations = [
            response.completion for response in judge_responses
            if response.status == STATUS_200 and response.completion
        ]
        return explanations

    @staticmethod
    def _validate_judge_trials(no_of_trials: int) -> None:
        if no_of_trials <= 0 or no_of_trials % 2 == 0:
            raise EvaluationError(internal_code=ErrorCode.INVALID_LLM_JUDGE_RESULT_ERROR.value,
                                  message="Number of judge trials must be a positive odd integer.")
