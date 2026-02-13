from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from agent_inspect.models.metrics.metric_score import NumericalScore


class MultiSampleMetric(ABC):
    """
    Base abstract class for metrics that aggregate results across multiple samples
    or trials.

    Concrete subclasses should implement logic that combines multiple
    ``NumericalScore`` objects into a single aggregated score.

    :param config: Optional configuration dictionary for metric initialization.
        Defaults to ``None``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def compute(
            self,
            scorer_results: List[NumericalScore],
    ):
        """
        Computes an aggregated metric score from multiple scorer results.

        This method is intended to be implemented by concrete subclasses that define
        how multiple trial-level or sample-level ``NumericalScore`` objects should be
        combined (for example, pass@k-style metrics).

        :param scorer_results: A list of
            :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` objects
            produced by scorer metrics, one per trial or sample.
        :return: A
            :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore` object
            containing the aggregated result.
        """
        ...
