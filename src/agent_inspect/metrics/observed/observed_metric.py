from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from agent_inspect.models.metrics.agent_trace import TurnTrace
from agent_inspect.models.metrics.metric_score import NumericalScore


class ObservedMetric(ABC):
    """
    This is a base abstract class that should be extended for actual implementations.

    :param config: configuration for metric initialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace],
    ) -> NumericalScore:
        """
        This is an abstract method and should be implemented in a concrete class.
        
        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object.
        """
        ...
