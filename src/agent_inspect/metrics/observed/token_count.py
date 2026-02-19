from abc import abstractmethod
from typing import Optional, Dict, Any, List

from agent_inspect.metrics.constants import INPUT_TOKEN_CONSUMPTION, OUTPUT_TOKEN_CONSUMPTION, REASONING_TOKEN_CONSUMPTION
from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.metrics.observed.observed_metric import ObservedMetric
from agent_inspect.models.metrics.agent_trace import TurnTrace
from agent_inspect.models.metrics.metric_score import NumericalScore


class TokenConsumptionMetric(ObservedMetric):
    """
    ObservedMetric to measure the token consumption responses per evaluation sample. 
    Initialise an instance of TokenConsumptionMetric.

    :param config: Configuration for token consumption metric initialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @abstractmethod
    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace],
    ) -> NumericalScore:
        """
        This is an abstract method and should be implemented in a concrete class.
        Calculate the token consumption by the agent.

        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the token consumption count.
        """
        ...

    @staticmethod
    def evaluate_by_field(agent_turn_traces: List[TurnTrace], field: str):
        total_token_count = 0
        for turn_trace in agent_turn_traces:
            if not turn_trace.steps:
                raise InvalidInputValueError(internal_code=ErrorCode.MISSING_VALUE.value,
                                             message=f"Turn: {turn_trace.id} has no steps.")
            for step in turn_trace.steps:
                total_token_count += getattr(step, field) if getattr(step, field) is not None else 0
        return NumericalScore(score=total_token_count)

class InputTotalTokenCount(TokenConsumptionMetric):
    """
    Metric to measure the input token consumption by the agent.

    :param config: Configuration for input token consumption metric initialization.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)


    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace],
    ) -> NumericalScore:
        """
        Calculate the input token consumption by the agent.

        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the total input token consumption count.        
        """
        return TokenConsumptionMetric.evaluate_by_field(agent_turn_traces, INPUT_TOKEN_CONSUMPTION)


class OutputTotalTokenCount(TokenConsumptionMetric):
    """
    Metric to measure the output token consumption by the agent.

    :param config: Configuration for output token consumption metric initialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace],
    ) -> NumericalScore:
        """
        Calculate the output token consumption by the agent.

        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the total output token consumption count.        
        """        
        return TokenConsumptionMetric.evaluate_by_field(agent_turn_traces, OUTPUT_TOKEN_CONSUMPTION)


class ReasoningTotalTokenCount(TokenConsumptionMetric):
    """
    Metric to measure the reasoning token consumption by the agent.

    :param config: Configuration for reasoning token consumption metric initialization.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace],
    ) -> NumericalScore:
        """
        Calculate the reasoning token consumption by the agent.

        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the total reasoning token consumption count.        
        """
        return TokenConsumptionMetric.evaluate_by_field(agent_turn_traces, REASONING_TOKEN_CONSUMPTION)

class TotalTokenConsumption(TokenConsumptionMetric):
    """
    Metric to measure the total token consumption consisting of input, output, reasoning tokens by the agent.

    :param config: Configuration for total token consumption metric initialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.input_total_token_count_metric = InputTotalTokenCount(config)
        self.output_total_token_count_metric = OutputTotalTokenCount(config)
        self.reasoning_total_token_count_metric = ReasoningTotalTokenCount(config)

    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace],
    ) -> NumericalScore:
        """
        Calculate the total token consumption by the agent.

        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the total token consumption count.        
        """
        total_token_count = self.input_total_token_count_metric.evaluate(agent_turn_traces).score
        total_token_count += self.output_total_token_count_metric.evaluate(agent_turn_traces).score
        total_token_count += self.reasoning_total_token_count_metric.evaluate(agent_turn_traces).score
        return NumericalScore(score=total_token_count)


