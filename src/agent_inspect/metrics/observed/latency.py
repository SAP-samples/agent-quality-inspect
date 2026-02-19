from abc import abstractmethod
from typing import Optional, Dict, Any, List

from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.metrics.observed.observed_metric import ObservedMetric
from agent_inspect.models.metrics.agent_trace import TurnTrace
from agent_inspect.models.metrics.metric_score import NumericalScore


class LatencyMetric(ObservedMetric):
    """
    Abstract class which should be extended for actual implementation of latency metric.
    Initialise an instance of LatencyMetric.

    :param config: Configuration for latency metric initialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @abstractmethod
    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace]
    ) -> NumericalScore:
        """
        This is an abstract method and should be implemented in a concrete class.
        Calculate the latency of the agent's response.

        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the latency score (float).
        """
        ...

class TotalLatency(LatencyMetric):
    """
    ObservedMetric to measure the total latency of agent responses per evaluation sample.

    :param config: Configuration for total latency metric initialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace],
    ) -> NumericalScore:
        """
        Calculate the total latency in ms of the agent responses.

        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the total latency score in ms (float).
        """
        total_latency = 0.0
        turns_missing_latency = []
        for turn in agent_turn_traces:
            if turn.latency_in_ms is None:
                turns_missing_latency.append(turn.id)
            else:
                total_latency += turn.latency_in_ms
        if turns_missing_latency:
            raise InvalidInputValueError(
                internal_code=ErrorCode.MISSING_VALUE.value,
                message=f"Turn(s): {', '.join(turns_missing_latency)} are missing latency values."
            )
        return NumericalScore(score=round(total_latency, 4))

class AverageLatency(LatencyMetric):
    """
    ObservedMetric to measure the average latency in ms of agent responses (per turn) per evaluation sample.

    :param config: Configuration for average latency metric initialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace],
    ):
        """
        Calculate the average latency of the agent's response.
        
        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the average latency score in ms per turn (float).
        """
        total_latency_metric = TotalLatency(self.config)
        total_latency = total_latency_metric.evaluate(agent_turn_traces).score
        return NumericalScore(score=round(total_latency/len(agent_turn_traces), 4) if len(agent_turn_traces) > 0 else 0.0)