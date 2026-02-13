from abc import abstractmethod

from typing import Any, Dict, Optional

from agent_inspect.metrics.scorer.metric import Metric
from agent_inspect.clients.llm_client import LLMClient
from agent_inspect.models.metrics.agent_trace import AgentDialogueTrace
from agent_inspect.models.metrics.agent_data_sample import EvaluationSample


class LLMBasedMetric(Metric):
    """
    This is a base abstract class that should be extended for actual implementations.

    :param llm_client: the client which allows connection to the LLM-as-a-judge model for evaluation.
    :param config: configuration for metric initialization. Default to ``None``.
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm_client = llm_client

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

    @staticmethod
    def get_turn_groupings_from_traces(agent_trace, turns_to_run):
        turns_groupings = []
        for i in range(turns_to_run):
            turns_groupings.append(agent_trace.turns[:i + 1])
        return turns_groupings
