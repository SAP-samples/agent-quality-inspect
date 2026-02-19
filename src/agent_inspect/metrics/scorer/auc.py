import sklearn

from typing import Any, Optional, Dict, List

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import EvaluationError, InvalidInputValueError
from agent_inspect.metrics.scorer.llm_based_metric import LLMBasedMetric
from agent_inspect.metrics.scorer.progress import ProgressScoresThroughTurns
from agent_inspect.models.metrics.agent_data_sample import EvaluationSample
from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace
from agent_inspect.models.metrics.metric_score import NumericalScore


class AUC(LLMBasedMetric):
    """
    Metric to calculate the area under the progress curve produced by :obj:`~agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns` class.  For computing AUC, the discrete progress values are treated as a continuous, monotonically increasing function obtained via linear interpolation.

    .. math::

        AUC = \int_{0}^{T} p(t) \ dt, 

    where :math:`T` is the maximum turns of a conversation and
    :math:`p(t) := progress(i, G_i, \\tau_i[1:t])` denotes the progress at turn :math:`t` (refer to the documentation on :obj:`~agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns`). 
    
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
        Returns the value of area under the progress scores curve. Calls the :obj:`agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns.evaluate` method underneath.

        :param agent_trace: a :obj:`~agent_inspect.models.metrics.agent_trace.AgentDialogueTrace` object constructed with the agent trajectory information for a given data sample.
        :param evaluation_data_sample: a :obj:`~agent_inspect.models.metrics.agent_data_sample.EvaluationSample` object representing a data sample in the evaluation data set.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing AUC score, sub scores consisting of progress scores at every turn, and judge explanations.

        Example:

        >>> from agent_inspect.metrics.scorer import AUC
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, MAX_TURNS, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> metric = AUC(
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
        progress_scores = progress_metric.evaluate(agent_trace, evaluation_data_sample)
        
        if not progress_scores:
            raise EvaluationError(internal_code=ErrorCode.EMPTY_PROGRESS_SCORE.value, message="No progress score produced for AUC calculation.")
        n = len(progress_scores)
        x = [i / (n - 1) if n > 1 else 0 for i in range(n)]
        progress_rates = [progress_score.score for progress_score in progress_scores]
        progress_explanations = [progress_score.explanations for progress_score in progress_scores]
        sub_scores = {}
        for idx, progress in enumerate(progress_rates):
            sub_scores[f"Turn_{idx + 1}_progress_score"] = progress

        return NumericalScore(score=round(sklearn.metrics.auc(x, progress_rates), 4), explanations=progress_explanations, sub_scores=sub_scores)

    @staticmethod
    def get_auc_score_from_progress_scores(progress_scores: List[NumericalScore]) -> NumericalScore:
        '''
        Computes the value of area under the progress scores curve given a list of progress scores at every conversational turn as input. The list of progress scores are obtained from :obj:`agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns.evaluate` method. 
        
        :param progress_scores: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.metric_score.NumericalScore`]  object storing a list of progress scores at every conversational turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the AUC score.

        Example:

        >>> from agent_inspect.metrics.scorer import ProgressScoresThroughTurns, AUC
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, INCLUDE_VALIDATION_RESULTS, MAX_TURNS, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> progress_turns_metric = ProgressScoresThroughTurns(
        ...     llm_client=client,
        ...     config={
        ...        MAX_TURNS: 8,
        ...        INCLUDE_VALIDATION_RESULTS: True,
        ...        INCLUDE_JUDGE_EXPLANATION: True,
        ...        OPTIMIZE_JUDGE_TRIALS: False
        ...    }
        ... )
        >>> progress_rates = progress_turns_metric.evaluate(
        ...     agent_trace=agent_trace,
        ...     evaluation_data_sample=data_sample
        ... )
        >>> auc_metric = AUC(llm_client=client)
        >>> auc_metric_result = auc_metric.get_auc_score_from_progress_scores(progress_rates)   
        >>> print(auc_metric_result.score)     
        '''
        
        if not progress_scores:
            raise InvalidInputValueError(internal_code=ErrorCode.EMPTY_PROGRESS_SCORE.value,
                                         message="No progress score provided for AUC calculation.")
        scores = [ps.score for ps in progress_scores]
        n = len(scores)
        x = [i / (n - 1) if n > 1 else 0 for i in range(n)]
        try:
            auc_score = round(sklearn.metrics.auc(x, scores), 4)
        except Exception as e:
            raise EvaluationError(internal_code=ErrorCode.AUC_CALCULATION_ERROR.value,
                                  message=f"Error calculating AUC: {str(e)}")
        return NumericalScore(score=auc_score)
