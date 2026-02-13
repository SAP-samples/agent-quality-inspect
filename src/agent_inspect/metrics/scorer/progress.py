import copy
import asyncio
from abc import abstractmethod

from typing import Optional, Dict, Any, List

from agent_inspect.exception.error_codes import ErrorCode

from agent_inspect.exception import InvalidInputValueError

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.scorer.llm_based_metric import LLMBasedMetric
from agent_inspect.models.metrics.agent_data_sample import EvaluationSample, SubGoal
from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace
from agent_inspect.models.metrics.metric_score import NumericalScore
from agent_inspect.metrics.constants import MAX_TURNS, MAX_TURNS_DEFAULT
from agent_inspect.metrics.utils.metrics_utils import get_config_or_default
from agent_inspect.metrics.constants import INCLUDE_VALIDATION_RESULTS
from agent_inspect.metrics.validator.subgoal_completion import SubGoalCompletionValidator
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult


class ProgressBasedMetric(LLMBasedMetric):
    """
    Abstract class which should be extended for actual implementations of progress metrics.

    :param llm_client: the client which allows connection to the llm-as-a-judge model for evaluations.
    :param config: configuration for progress metric initialization. Default to ``None``.
    """
    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, config)

    @abstractmethod
    def evaluate(
            self,
            agent_trace: AgentDialogueTrace,
            evaluation_data_sample: EvaluationSample,
    ):
        """
        This is an abstract method and should be implemented in a concrete class.

        :param agent_trace: a :obj:`~agent_inspect.models.metrics.agent_trace.AgentDialogueTrace` object constructed with the agent trajectory information for a given data sample.
        :param evaluation_data_sample: a :obj:`~agent_inspect.models.metrics.agent_data_sample.EvaluationSample` object representing a data sample in the evaluation data set.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object or a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.metric_score.NumericalScore`] object.
        """
        ...


class ProgressScore(ProgressBasedMetric):
    """
    Metric to calculate agent's progress for a given task sample based on the proportion of subgoals completed.
    Current metric supports only static conversation where the user utterances are predetermined.

    .. math::

        progress(i, G_i, \\tau_i)=\\frac{1}{|G_i|} \sum_{j=1}^{|G_i|} LLM_{judge}(i, g_{i, j}, \\tau_i ),

    where :math:`LLM_{judge}(\\cdot)` is the output from the LLM-as-a-judge,
    :math:`G_i= \\{ g_{i, 1}, ..., g_{i, j}, ...,  g_{i, |G_i|} \\}` is the set of subgoals a.k.a grading notes for task sample :math:`i`, and
    :math:`\\tau_i` is the agent trajectory for the entire conversation consisting of tool calls, agent responses, and user inputs.

    :param llm_client: the client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: Default to ``None``. Configuration options:

        - **templates_subgoal**: a user provided LLM-as-a-judge template which will be sent to the judge with user inputs, subgoal, trajectory, and agent responses. If this is not provided, the default template for static single-turn or static multi-turn conversation will be used.
        - **num_judge_trials**: the number of LLM-as-a-judge runs. Default to ``5``. A majority vote is used when the number of LLM-as-a-judge runs is set to a value larger than 1.
        - **include_judge_explanation**: a :obj:`~typing.bool` flag to indicate whether the output should also return judge explanations. Default to ``False``.
        - **include_entire_prompt_in_validation_result**: a :obj:`~typing.bool` flag to indicate whether to include the entire prompt sent to LLM-as-a-judge in the SubGoalValidationResult. Use this in the debugging tool UI for display. Default to ``False``.
        - **optimize_judge_trials**: a :obj:`~typing.bool` flag to indicate whether to use optimized judge runs when doing a majority vote. Default to ``False``.
        - **max_retry_judge_trials**: an :obj:`~typing.int` value indicating the maximum number of retry attempts for each judge trial in case of errors related to LLM as a judge. Default to ``5``. This will be ignored if ``optimize_judge_trials`` is set to ``True``.

    """
    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, config)

    def evaluate(
            self,
            agent_trace: AgentDialogueTrace,
            evaluation_data_sample: EvaluationSample,
    ):
        """
        Returns a progress score given the agent trace and the evaluation data sample.

        :param agent_trace: a :obj:`~agent_inspect.models.metrics.agent_trace.AgentDialogueTrace` object constructed with the agent trajectory information for a given data sample.
        :param evaluation_data_sample: a :obj:`~agent_inspect.models.metrics.agent_data_sample.EvaluationSample` object representing a data sample in the evaluation data set.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing progress score and judge explanations.

        Example:

        >>> from agent_inspect.metrics.scorer import ProgressScore
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> metric = ProgressScore(
        ...     llm_client=client,
        ...     config={INCLUDE_JUDGE_EXPLANATION: True, OPTIMIZE_JUDGE_TRIALS: False}
        ... )
        >>> metric_result = metric.evaluate(
        ...     agent_trace=agent_trace,
        ...     evaluation_data_sample=data_sample
        ... )
        >>> print(metric_result.score)

        """
        sub_goals = evaluation_data_sample.sub_goals

        validation_results = []
        explanations = []
        goal_completion_validator = SubGoalCompletionValidator(llm_client=self.llm_client, config=self.config)

        turns_to_run = len(agent_trace.turns)
        turns_groupings = ProgressBasedMetric.get_turn_groupings_from_traces(agent_trace, turns_to_run)

        for idx, turns_grouping in enumerate(turns_groupings):
            # print(f"Evaluating at turn: {idx + 1} with previous turns as context")
            current_subgoals = ProgressScore.get_turn_subgoals(sub_goals, idx)
            for current_subgoal in current_subgoals:
                validation_result = asyncio.run(goal_completion_validator.validate(turn_traces=turns_grouping, sub_goal=current_subgoal))
                # print(f"Result: {validation_result.is_completed}")
                # TODO: store subgoal turn and type for judge_explanation incase subgoal.details is not unique
                explanations.append({current_subgoal.details: validation_result.explanations})
                validation_results.append(validation_result)
        progress_score = ProgressScore.get_progress_score_from_validation_results(validation_results)
        progress_score.explanations = explanations
        progress_score.validation_results = validation_results
        return progress_score

    @staticmethod
    def get_progress_score_from_validation_results(validation_results: List[SubGoalValidationResult]) -> NumericalScore:
        scores = []
        if not validation_results:
            raise InvalidInputValueError(internal_code=ErrorCode.EMPTY_VALIDATION_RESULT.value,
                                         message="No validation result present to aggregate for progress score.")
        for validation_result in validation_results:
            if validation_result.is_completed:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return NumericalScore(score=round(sum(scores) / len(scores), 4))

    @staticmethod
    def get_turn_subgoals(sub_goals: list[SubGoal], turn_index: int):
        per_turn_subgoals = []
        for sub_goal in sub_goals:
            if sub_goal.turn == turn_index:
                per_turn_subgoals.append(sub_goal)
        return per_turn_subgoals


class ProgressScoresThroughTurns(ProgressBasedMetric):
    """
    Metric to calculate agent's progress at every conversational turn (up to the final conversation turn :math:`T`) for a given task sample based on the proportion of subgoals completed. Subgoals that are completed at the current or previous turns are not evaluated again in the subsequent turns. The metric assumes previously completed subgoals which are milestones cannot be undone.

    For every conversational turn :math:`t` up to the final turn :math:`T`, the agent's progress at turn :math:`t` is computed as follows:

    .. math::

        progress(i, G_i, \\tau_i[1:t])=\\frac{1}{|G_i|} \sum_{j=1}^{|G_i|} LLM_{judge}(i, g_{i, j}, \\tau_i[1:t]), 
    
    where :math:`LLM_{judge}(\\cdot)` is the output from the LLM-as-a-judge,
    :math:`G_i= \\{ g_{i, 1}, ..., g_{i, j}, ...,  g_{i, |G_i|} \\}` is the set of subgoals a.k.a grading notes for task sample :math:`i`, and
    :math:`\\tau_i[1:t]` is the segment of agent trajectory from the first turn up to turn :math:`t` consisting of tool calls, agent responses, and user inputs.

    :param llm_client: the client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: Default to ``None``. Configuration options:

        - **templates_subgoal**: a user provided LLM-as-a-judge template which will be sent to the judge with user task, subgoal, trajectory, user utterances, and agent responses. If this is not provided, the default template for dynamic conversation will be used.
        - **num_judge_trials**: the number of LLM-as-a-judge runs. Default to ``5``. A majority vote is used when the number of LLM-as-a-judge runs is set to a value larger than 1.
        - **include_judge_explanation**: a :obj:`~typing.bool` flag to indicate whether the output should also return judge explanations. Default to ``False``.
        - **include_validation_results**: a :obj:`~typing.bool` flag to indicate whether the output should also return a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.validation_result.SubGoalValidationResult`]. This is used later for error analysis. Default to ``False``.
        - **include_entire_prompt_in_validation_result**: a :obj:`~typing.bool` flag to indicate whether to include the entire prompt sent to LLM-as-a-judge in the SubGoalValidationResult. Use this in the debugging tool UI for display. Default to ``False``.
        - **optimize_judge_trials**: a :obj:`~typing.bool` flag to indicate whether to use optimized judge runs when doing a majority vote. This needs to be set to ``False`` in order to perform error analysis later. Default to ``False``.
        - **max_retry_judge_trials**: an :obj:`~typing.int` value indicating the maximum number of retry attempts for each judge trial in case of errors related to LLM as a judge. Default to ``5``. This will be ignored if ``optimize_judge_trials`` is set to ``True``.
        - **max_turns**: Evaluate the agent up to max_turns conversation only.  For conversation shorter than ``max_turns``, the final progress score is populated up to ``max_turns``. Default to ``20``.

    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, config)

    def evaluate(
            self,
            agent_trace: AgentDialogueTrace,
            evaluation_data_sample: EvaluationSample,
    ):
        """
        Returns a list of progress scores at every turn until ``max_turns`` given the agent trace and the evaluation data sample.

        :param agent_trace: a :obj:`~agent_inspect.models.metrics.agent_trace.AgentDialogueTrace` object constructed with the agent trajectory information for a given data sample.
        :param evaluation_data_sample: a :obj:`~agent_inspect.models.metrics.agent_data_sample.EvaluationSample` object representing a data sample in the evaluation data set.
        :return: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.metric_score.NumericalScore`]  object storing a list of progress scores at every turn until ``max_turns``.

        Example:

        >>> from agent_inspect.metrics.scorer import ProgressScoresThroughTurns
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, INCLUDE_VALIDATION_RESULTS, MAX_TURNS, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> metric = ProgressScoresThroughTurns(
        ...     llm_client=client,
        ...     config={
        ...        MAX_TURNS: 8,
        ...        INCLUDE_VALIDATION_RESULTS: True,
        ...        INCLUDE_JUDGE_EXPLANATION: True,
        ...        OPTIMIZE_JUDGE_TRIALS: False
        ...    }
        ... )
        >>> metric_result = metric.evaluate(
        ...     agent_trace=agent_trace,
        ...     evaluation_data_sample=data_sample
        ... )
        >>> print(metric_result) # print list of NumericalScore objects

        """
        include_validation_results = get_config_or_default(config=self.config, config_key=INCLUDE_VALIDATION_RESULTS, default=False)
        
        scores = []
        judge_explanations_turns=[]
   
        max_turn = get_config_or_default(config=self.config, config_key=MAX_TURNS, default=MAX_TURNS_DEFAULT)

        turns_to_run = min(max_turn, len(agent_trace.turns))
        turns_groupings = ProgressBasedMetric.get_turn_groupings_from_traces(agent_trace, turns_to_run)

        user_task = evaluation_data_sample.user_instruction

        remaining_goals = copy.deepcopy(evaluation_data_sample.sub_goals)
        completed_goals = []

        goal_completion_validator = SubGoalCompletionValidator(llm_client=self.llm_client, config=self.config)
        validation_results_completed: list[SubGoalValidationResult] = []
        full_validation_results = []

        for idx, turns_grouping in enumerate(turns_groupings):
            turn_validation_results = [] # stores results for the remaining subgoals
            # print(f"Evaluating with traces up to turn: {idx + 1}")
            for sub_goal in remaining_goals:
                validation_result = asyncio.run(goal_completion_validator.validate_dynamic(turn_traces=turns_grouping, sub_goal=sub_goal, user_instruction=user_task))
                if validation_result.is_completed:
                    completed_goals.append(sub_goal)
                turn_validation_results.append(validation_result)
            
            for goal in completed_goals:
                remaining_goals.remove(goal)
            completed_goals = []

            full_validation_results = validation_results_completed + turn_validation_results
            progress_turn = ProgressScore.get_progress_score_from_validation_results(full_validation_results)
            scores.append(round(progress_turn.score, 4))
            # TODO: store subgoal turn and type for judge_explanation incase subgoal.details is not unique
            judge_explanations = [{r.sub_goal.details: r.explanations} for r in full_validation_results]
            judge_explanations_turns.append(judge_explanations)
            validation_results_completed += [r for r in turn_validation_results if r.is_completed == True]

        if max_turn > turns_to_run :
            scores = scores + [scores[-1]] * (max_turn - turns_to_run)
            judge_explanations_turns = judge_explanations_turns + [judge_explanations_turns[-1]]* (max_turn - turns_to_run)

        score_objs = []
        for score, judge_exp in zip(scores, judge_explanations_turns):
            score_objs.append(NumericalScore(score=score, explanations=judge_exp))
        if include_validation_results and score_objs:
            score_objs[-1].validation_results = full_validation_results 
        return score_objs
