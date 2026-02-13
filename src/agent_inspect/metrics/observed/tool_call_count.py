from typing import Optional, Dict, Any, List

from agent_inspect.metrics.observed.observed_metric import ObservedMetric
from agent_inspect.models.metrics.agent_trace import TurnTrace
from agent_inspect.models.metrics.metric_score import NumericalScore


class ToolCallCount(ObservedMetric):
    """
    ToolCallCountMetric to measure the total number of tools called by the agent per evaluation sample.

    :param config: Configuration for ToolCallCountMetric initialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)


    def evaluate(
            self,
            agent_turn_traces: List[TurnTrace]
    ) -> NumericalScore:
        """
        Calculate the total number of tools called by the agent.

        :param agent_turn_traces: a :obj:`~typing.List` [:obj:`~agent_inspect.models.metrics.agent_trace.TurnTrace`] object constructed with the agent trajectory information from the first turn up to the current turn.
        :return: a :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object containing the total number of tool calls.        
        """

        tool_call_count = 0
        for turn_trace in agent_turn_traces:
            for step in turn_trace.steps:
                if step.tool and step.tool.strip():
                    tool_call_count += 1
        return NumericalScore(score=tool_call_count)

