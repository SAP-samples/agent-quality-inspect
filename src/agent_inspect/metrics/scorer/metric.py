from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace
from agent_inspect.models.metrics.agent_data_sample import EvaluationSample


class Metric(ABC):
    """
    This is a base abstract class that should be extended for actual implementations.

    :param config: configuration for metric initialization. Default to ``None``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def evaluate(
            self,
            agent_trace: AgentDialogueTrace,
            evaluation_data_sample: EvaluationSample,
    ):
        """
        This is an abstract method and should be implemented in a concrete class.

        :param agent_trace: a :obj:`~agent_inspect.models.agent_trace.AgentDialogueTrace` object constructed with the agent trajectory information for a given data sample.
        :param evaluation_data_sample: a :obj:`~agent_inspect.models.agent_data_sample.EvaluationSample` object representing a data sample in the evaluation data set.
        :return: a :obj:`~agent_inspect.models.metric_score.NumericalScore` object or a :obj:`~typing.List` [:obj:`~agent_inspect.models.metric_score.NumericalScore`] object.
        """
        ...