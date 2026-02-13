
from typing import Any, Optional, Dict, List

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import EvaluationError, InvalidInputValueError
from agent_inspect.metrics.scorer.progress import ProgressBasedMetric
from agent_inspect.metrics.scorer.progress import ProgressScoresThroughTurns
from agent_inspect.models.metrics.agent_data_sample import EvaluationSample
from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace
from agent_inspect.models.metrics.metric_score import NumericalScore


class PPT(ProgressBasedMetric):
    """
    Metric to calculate the progress-per-turn (PPT) of the progress scores produced by :obj:`~agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns` class.
    PPT metric is defined as the total increase in progress divided by the number of turns. It weights the increase in progress uniformly across the conversational turns. 

    .. math::

        PPT = \\frac{1}{T} \sum_{t=0}^{T-1} p(t+1)-p(t)= \\frac{p(T)}{T}, 

    where :math:`p(t) := progress(i, G_i, \\tau_i[1:t])` denotes the discrete progress at turn :math:`t` (refer to the documentation on :obj:`~agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns`), 
    :math:`T` is the minimum number of conversational turns to reach the final achieved progress :math:`p(T)`, and 
    :math:`p(0)=0`. 

    :param llm_client: the client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: Default to ``None``. Configuration options:

        - **templates_subgoal**: a user provided LLM-as-a-judge template which will be sent to the judge with user task, subgoal, trajectory, user utterances, and agent responses. If this is not provided, the default template for dynamic conversation will be used.
        - **num_judge_trials**: the number of LLM-as-a-judge runs. Default to ``5``. A majority vote is used when the number of LLM-as-a-judge runs is set to a value larger than 1.
        - **include_judge_explanation**: a :obj:`~typing.bool` flag to indicate whether the output should also return judge explanations. Default to ``False``.
        - **optimize_judge_trials**: a :obj:`~typing.bool` flag to indicate whether to use optimized judge runs when doing a majority vote. This needs to be set to ``False`` in order to perform error analysis later. Default to ``False``.
        - **max_retry_judge_trials**: an :obj:`~typing.int` value indicating the maximum number of retry attempts for each judge trial in case of errors related to LLM as a judge. Default to ``5``. This will be ignored if ``optimize_judge_trials`` is set to ``True``.
        - **max_turns**: Evaluate the agent up to max_turns conversation only.  For conversation shorter than ``max_turns``, the final progress score is populated up to ``max_turns``. Default to ``20``.
    """

    def __init__(self, llm_client: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, config)

    def evaluate(
            self,
            agent_trace: AgentDialogueTrace,
            evaluation_data_sample: EvaluationSample,
    ):
        """
        Returns the progress-per-turn (PPT) value of the list of progress scores. Calls the :obj:`agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns.evaluate` method underneath.

        :param agent_trace: a :obj:`~agent_inspect.models.metrics.agent_trace.AgentDialogueTrace` object constructed with the agent trajectory information for a given data sample.
        :param evaluation_data_sample: a :obj:`~agent_inspect.models.metrics.agent_data_sample.EvaluationSample` object representing a data sample in the evaluation data set.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing PPT score, sub scores consisting of progress scores at every turn, and judge explanations.

        Example:

        >>> from agent_inspect.metrics.scorer import PPT
        >>> from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, MAX_TURNS, OPTIMIZE_JUDGE_TRIALS
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096) # create client needed for LLM-based metric
        >>> metric = PPT(
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
            raise EvaluationError(internal_code=ErrorCode.EMPTY_PROGRESS_SCORE.value, message="No progress score produced for PPT calculation.")
        
        ppt_score = self.get_ppt_score_from_progress_scores(progress_scores)
        
        # Add sub_scores for individual turn progress scores
        progress_rates = [progress_score.score for progress_score in progress_scores]
        progress_explanations = [progress_score.explanations for progress_score in progress_scores]
        sub_scores = {}
        for idx, progress in enumerate(progress_rates):
            sub_scores[f"Turn_{idx + 1}_progress_score"] = progress
                
        return NumericalScore(score=round(ppt_score.score, 4), explanations=progress_explanations, sub_scores=sub_scores)
    
        
    @staticmethod
    def get_ppt_score_from_progress_scores(progress_scores: List[NumericalScore]) -> NumericalScore:
        '''
        Computes the progress-per-turn (PPT) value given a list of progress scores at every conversational turn as input. The list of progress scores are obtained from :obj:`agent_inspect.metrics.scorer.progress.ProgressScoresThroughTurns.evaluate` method. 

        :param progress_scores: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.metric_score.NumericalScore`]  object storing a list of progress scores at every conversational turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the PPT score.

        Example:

        >>> from agent_inspect.metrics.scorer import ProgressScoresThroughTurns, PPT
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
        >>> ppt_metric = PPT(llm_client=client)
        >>> ppt_metric_result = ppt_metric.get_ppt_score_from_progress_scores(progress_rates)   
        >>> print(ppt_metric_result.score)    
        '''
        
        if not progress_scores:
            raise InvalidInputValueError(internal_code=ErrorCode.EMPTY_PROGRESS_SCORE.value,
                                         message="No progress score provided for PPT calculation.")
        
        progress_list = [progress_score.score for progress_score in progress_scores]
        
        try:
            # Find the maximum value
            max_value = max(progress_list)
            
            # Find the first occurrence (earliest) of the maximum value
            earliest_max_index = None
            for i, value in enumerate(progress_list):
                if value == max_value:
                    earliest_max_index = i
                    break
            
            # Divide max value by (index + 1)
            ppt_score = max_value / (earliest_max_index + 1)
            
        except Exception as e:
            raise EvaluationError(internal_code=ErrorCode.PPT_CALCULATION_ERROR.value,
                                  message=f"Error calculating PPT: {str(e)}")
        
        return NumericalScore(score=ppt_score)