import asyncio
from typing import Optional, Dict, Any, List

from agent_inspect.exception.error_codes import ErrorCode

from agent_inspect.exception import InvalidInputValueError

from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.models.metrics.metric_score import NumericalScore
from agent_inspect.models.metrics.agent_data_sample import EvaluationSample
from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace
from agent_inspect.metrics.validator.tool_call_completion import ToolCallCompletionValidator
from agent_inspect.models.metrics.validation_result import ToolCallValidationResult
from agent_inspect.metrics.scorer.llm_based_metric import LLMBasedMetric


class ToolCorrectnessMetric(LLMBasedMetric):
    """
    Metric to calculate the correctness rate of tool calls made by an agent in its entire dialogue trace. The final score is computed as the ratio of correct tool calls to the total number of tool calls made.

    The tool correctness score :math:`\\text{tool_correctness}(i, T_i, \\tau_i)` for sample :math:`i` is defined as:

    .. math::

        \\text{tool_correctness}(i, T_i, \\tau_i) = \\frac{1}{N} \sum_{j=1}^N \\mathbb{I}(T_{i,j} \\text{ is called in } \\tau_i)
        
    where :math:`\\tau_i` refers to the agent trajectory for sample :math:`i`, :math:`T_i = \{T_{i,1}, T_{i,2}, \ldots, T_{i,N}\}` represents the set of :math:`N` expected tool calls for sample :math:`i`, and :math:`\\mathbb{I}(\\cdot)` is the indicator function that equals 1 if the :math:`j`-th tool call :math:`T_{i,j}` is correctly called by the agent, and 0 otherwise. 
    
    The correctness of each tool call is determined by validating the agent's tool call against the expected tool call using both exact match and LLM-as-a-judge approaches, depending on the configuration. Specifically, if an argument or parameter is set to :obj:`~agent_inspect.models.metrics.agent_data_sample.ToolInputParameter.value`, exact match is used; if set to :obj:`~agent_inspect.models.metrics.agent_data_sample.ToolInputParameter.check`, LLM-as-a-judge validates the correctness. The evaluation is performed across three dimensions: tool name, tool input arguments, and tool output. One tool call is considered correct only if all three dimensions are validated as correct.

    :param llm_client: The client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: Default to ``None``. Configuration options:

        - **num_judge_trials**: the number of LLM-as-a-judge runs. Default to ``5``. A majority vote is used when the number of LLM-as-a-judge runs is set to a value larger than 1.
        - **include_judge_explanation**: a :obj:`~typing.bool` flag to indicate whether the output should also return judge explanations. Default to ``False``.
    """
    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm_client, config)
    
    
    def evaluate(
        self,
        agent_trace: AgentDialogueTrace,
        evaluation_data_sample: EvaluationSample,
    ):
        """
        Returns a tool correctness score given the agent trace and the evaluation data sample. 
        
        :param agent_trace: Agent Trace object constructed with the traces produced by the data sample.
        :param evaluation_data_sample: Data Sample object that represents a data sample in the evaluation data set.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` containing the tool correctness score (:obj:`~typing.float`) and explanations.

        Example:
        
        >>> from agent_inspect.metrics.scorer import ToolCorrectnessMetric
        >>> from agent_inspect.metrics.constants import NUM_JUDGE_TRIALS 
        >>> from agent_inspect.clients import AzureOpenAIClient
        >>>
        >>> data_sample=load_data_sample(sample_path) # Load data sample
        >>> agent_trace = load_agent_trace(trace_file_path) # Load agent trajectory information
        >>> client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096)  # create client needed for LLM-based metric
        >>> metric = ToolCorrectnessMetric(
        ...     llm_client=client,
        ...     config={
        ...         NUM_JUDGE_TRIALS: 5
        ...     }
        ... )
        >>> tool_correctness_score = metric.evaluate(
        ...     agent_trace=agent_trace,
        ...     evaluation_data_sample=data_sample
        ... )
        >>> print(tool_correctness_score.score)
        """
        tool_call_validator = ToolCallCompletionValidator(llm_client=self.llm_client, config=self.config)
        turns_to_run = len(agent_trace.turns)
        turns_groupings = LLMBasedMetric.get_turn_groupings_from_traces(agent_trace, turns_to_run)
        
        validation_results = []
        explanations = []
        for idx, turns_grouping in enumerate(turns_groupings):
            for expected_tool_call in evaluation_data_sample.expected_tool_calls:
                if expected_tool_call.turn == idx:
                    validation_result = asyncio.run(tool_call_validator.validate(agent_trace_turns = turns_grouping, expected_tool_call = expected_tool_call))
                    validation_results.append(validation_result)
                    # TODO: need to store the entire information of `expected_tool_call` incase of duplicated GT across turns (beyond MVS)
                    explanations.append({
                        expected_tool_call.tool: validation_result.explanations
                    })

        # print(f"Total tool calls evaluated: {len(validation_results)}, {validation_results}")
        tool_correctness_score = ToolCorrectnessMetric.get_tool_correctness_score_from_validation_results(validation_results)
        tool_correctness_score.explanations = explanations
        return tool_correctness_score

    @staticmethod
    def get_tool_correctness_score_from_validation_results(validation_results: List[ToolCallValidationResult]) -> NumericalScore:
        scores = []
        if not validation_results:
            raise InvalidInputValueError(internal_code=ErrorCode.EMPTY_VALIDATION_RESULT.value, message="No validation results present to aggregate for tool correctness score.")
        for validation_result in validation_results:
            if validation_result.is_completed:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return NumericalScore(score=round(sum(scores) / len(scores), 4))
        
