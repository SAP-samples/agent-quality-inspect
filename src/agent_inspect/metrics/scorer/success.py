from abc import abstractmethod
from typing import Optional, Any, Dict, List

from agent_inspect.exception.error_codes import ErrorCode

from agent_inspect.exception import InvalidInputValueError, EvaluationError

from agent_inspect.models.metrics.validation_result import SubGoalValidationResult

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.metrics.scorer.llm_based_metric import LLMBasedMetric
from agent_inspect.metrics.scorer.progress import ProgressScore, ProgressScoresThroughTurns
from agent_inspect.models.metrics.agent_data_sample import EvaluationSample
from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace
from agent_inspect.models.metrics.metric_score import NumericalScore


class SuccessBasedMetric(LLMBasedMetric):
    """
    Abstract class which should be extended for actual implementations of success metrics.

    :param llm_client: the client which allows connection to the llm-as-a-judge model for evaluations.
    :param config: configuration for success metric initialization. Default to ``None``.
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

    @staticmethod
    def get_success_score_from_validation_results(validation_results: List[SubGoalValidationResult]) -> NumericalScore:
        """
        Aggregates a list of SubGoalValidationResult objects to compute a success score. Success score is 1 if all the validation results indicate success, and 0 otherwise.

        :param validation_results: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.validation_result.SubGoalValidationResult`] object containing the result of subgoal validations.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing success score and sub scores consisting of progress score.
        """
        if not validation_results:
            raise InvalidInputValueError(internal_code=ErrorCode.EMPTY_VALIDATION_RESULT.value,
                                         message="No validation result present to aggregate for success score.")
        progress_score = ProgressScore.get_progress_score_from_validation_results(validation_results)
        if progress_score.score == 1:
            success_score = 1
        else:
            success_score = 0
        return NumericalScore(score=success_score, sub_scores={
            "progress_score": progress_score.score
        })

    @staticmethod
    def get_success_score_from_progress_score(progress_score: NumericalScore) -> NumericalScore:
        """
        Computes the success score given the progress score. Success score is 1 if the progress score is 1, and 0 otherwise.

        :param progress_score: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the progress score
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing success score and sub scores consisting of progress score.
        
        Example:

        >>> from agent_inspect.metrics.scorer import ProgressScore, SuccessBasedMetric
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> progress_metric = ProgressScore(
        ...     llm_client=client,
        ...     config={INCLUDE_JUDGE_EXPLANATION: True, OPTIMIZE_JUDGE_TRIALS: False}
        ... )
        >>> progress_metric_result = progress_metric.evaluate(
        ...     agent_trace=agent_trace,
        ...     evaluation_data_sample=data_sample
        ... )
        >>> success_score = SuccessBasedMetric.get_success_score_from_progress_score(progress_metric_result)
        >>> print(success_score)
        """
        if progress_score.score == 1:
            success_score = 1
        else:
            success_score = 0
        return NumericalScore(score=success_score, explanations=progress_score.explanations, sub_scores={
            "progress_score": progress_score.score
        })


class SuccessScore(SuccessBasedMetric):
    """
    Metric to calculate agent's success rate for a given task sample based on the agent's progress. Current metric supports only static conversation where the user utterances are predetermined.

    .. math::

        success(i, G_i, \\tau_i) = 1 \  \\mathrm{if} \  progress(i, G_i, \\tau_i)=1, \  \\mathrm{and} \ 0  \ \\mathrm{otherwise}, 

    where :math:`progress(i, G_i, \\tau_i)` is the progress score of the agent (refer to the documentation on :obj:`~agent_inspect.metrics.scorer.progress.ProgressScore`),
    :math:`G_i` is the set of subgoals a.k.a grading notes for task sample :math:`i`, and
    :math:`\\tau_i` is the agent trajectory for the entire conversation consisting of tool calls, agent responses, and user inputs.        

    :param llm_client: the client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: Default to ``None``. Configuration options:

        - **templates_subgoal**: a user provided LLM-as-a-judge template which will be sent to the judge with user inputs, subgoal, trajectory, and agent responses. If this is not provided, the default template for static single-turn or static multi-turn conversation will be used.
        - **num_judge_trials**: the number of LLM-as-a-judge runs. Default to ``5``. A majority vote is used when the number of LLM-as-a-judge runs is set to a value larger than 1.
        - **include_judge_explanation**: a :obj:`~typing.bool` flag to indicate whether the output should also return judge explanations. Default to ``False``.
        - **optimize_judge_trials**: a :obj:`~typing.bool` flag to indicate whether to use optimized judge runs when doing a majority vote. Default to ``False``.
        - **max_retry_judge_trials**: an :obj:`~typing.int` value indicating the maximum number of retry attempts for each judge trial in case of errors related to LLM as a judge. Default to ``5``. This will be ignored if ``optimize_judge_trials`` is set to ``True``.

    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialise an instance of SuccessScore Metric.

        :param llm_client: The client which allows connection to the llm-as-a-judge model for evaluations.
        :param config:
            templates_subgoal: user provided llm-as-a-judge template which will be sent to the judge with user input, subgoal, trajectories, agent response.
            num_judge_trials: the number of evaluations done on the same subgoal. Used in majority vote mechanism to get final result by reducing judge inconsistency. Default to 5.
            include_judge_explanation: flag to indicate whether the output should contain all judge explanations. Default to False.
        """
        super().__init__(llm_client, config)

    def evaluate(
            self,
            agent_trace: AgentDialogueTrace,
            evaluation_data_sample: EvaluationSample,
    ):
        """
        Returns a success score given the agent trace and the evaluation data sample.
        Calls the :obj:`agent_inspect.metrics.scorer.progress.ProgressScore.evaluate` and :obj:`agent_inspect.metrics.scorer.success.SuccessBasedMetric.get_success_score_from_progress_score` methods underneath.

        :param agent_trace: Agent Trace object constructed with the traces produced by the data sample.
        :param evaluation_data_sample: Data Sample object that represents a data sample in the evaluation data set.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing success score, sub scores consisting of progress score, and judge explanations.

        Example:

        >>> from agent_inspect.metrics.scorer import SuccessScore
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> metric = SuccessScore(
        ...     llm_client=client,
        ...     config={INCLUDE_JUDGE_EXPLANATION: True, OPTIMIZE_JUDGE_TRIALS: False}
        ... )
        >>> metric_result = metric.evaluate(
        ...     agent_trace=agent_trace,
        ...     evaluation_data_sample=data_sample
        ... )
        >>> print(metric_result.score)
        """
        progress_metric = ProgressScore(self.llm_client, self.config)
        result = progress_metric.evaluate(agent_trace, evaluation_data_sample)
        return SuccessBasedMetric.get_success_score_from_progress_score(result)


class SuccessScoreFinalTurn(SuccessBasedMetric):
    """
    Metric to calculate agent's success score  for a given task sample based on the agent's progress at the final conversational turn :math:`T`. 

    .. math::

        success(i, G_i, \\tau_i[1:T]) = 1 \  \\mathrm{if} \  progress(i, G_i, \\tau_i[1:T])=1, \  \\mathrm{and} \ 0  \ \\mathrm{otherwise}, 

    where :math:`progress(i, G_i, \\tau_i[1:T])` is the progress score of the agent at the final conversation turn :math:`T` (refer to the documentation on :obj:`~agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns`),
    :math:`G_i` is the set of subgoals a.k.a grading notes for task sample :math:`i`, and
    :math:`\\tau_i[1:T]` is the segment of agent trajectory from the first turn up to final turn :math:`T` consisting of tool calls, agent responses, and user inputs.      

    :param llm_client: the client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: Default to ``None``. Configuration options:

        - **templates_subgoal**: a user provided LLM-as-a-judge template which will be sent to the judge with user task, subgoal, trajectory, user utterances, and agent responses. If this is not provided, the default template for dynamic conversation will be used.
        - **num_judge_trials**: the number of LLM-as-a-judge runs. Default to ``5``. A majority vote is used when the number of LLM-as-a-judge runs is set to a value larger than 1.
        - **include_judge_explanation**: a :obj:`~typing.bool` flag to indicate whether the output should also return judge explanations. Default to ``False``.
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
        Returns a success score at the final conversational turn :math:`T` given the agent trace and the evaluation data sample.
        Calls the :obj:`agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns.evaluate` and :obj:`agent_inspect.metrics.scorer.success.SuccessBasedMetric.get_success_score_from_progress_score` methods underneath.

        :param agent_trace: Agent Trace object constructed with the traces produced by the data sample.
        :param evaluation_data_sample: Data Sample object that represents a data sample in the evaluation data set.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing success score at final turn :math:`T`, sub scores consisting of progress scores at every turn, and judge explanations.

        Example:

        >>> from agent_inspect.metrics.scorer import SuccessScoreFinalTurn
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, MAX_TURNS, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> metric = SuccessScoreFinalTurn(
        ...     llm_client=client,
        ...     config={
        ...        MAX_TURNS: 8,
        ...        INCLUDE_JUDGE_EXPLANATION: True,
        ...        OPTIMIZE_JUDGE_TRIALS: False
        ...    }
        ... )
        >>> metric_result = metric.evaluate(
        ...     agent_trace=agent_trace,
        ...     evaluation_data_sample=data_sample
        ... )
        >>> print(metric_result.score)
        """

        progress_metric = ProgressScoresThroughTurns(self.llm_client, self.config)
        results = progress_metric.evaluate(agent_trace, evaluation_data_sample)
        if results is None or len(results) == 0:
            raise EvaluationError(internal_code=ErrorCode.EMPTY_PROGRESS_SCORE.value, message="Progress score evaluation returned None or empty list.")
        success_score = SuccessBasedMetric.get_success_score_from_progress_score(results[-1])
        success_explanation = results[-1].explanations
        sub_scores = {}
        for idx, result in enumerate(results):
            sub_scores[f"Turn_{idx + 1}_progress_score"] = result.score
        success_score.explanations = success_explanation
        success_score.sub_scores = sub_scores

        return success_score
