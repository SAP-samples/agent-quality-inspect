import math
from typing import Optional, Dict, Any, List

from agent_inspect.exception import EvaluationError, ErrorCode
from agent_inspect.models.metrics.metric_score import NumericalScore
from agent_inspect.metrics.utils.metrics_utils import get_config_or_default, validate_inputs_for_pass_k_initialisation
from agent_inspect.metrics.constants import K_VALUE, NO_OF_TRIALS
from agent_inspect.metrics.multi_samples.multi_sample_metric import MultiSampleMetric



class PassAtK(MultiSampleMetric):
    """
    Metric to calculate pass@k: the probability that at least one of k randomly sampled trials is successful.

    .. math::

        pass@k = 1 - \\frac{\\binom{n-s}{k}}{\\binom{n}{k}}

    where:
        - n: total number of trials
        - s: number of successful trials
        - k: number of samples drawn

    :param k: Number of samples to draw (default: None, must be set before evaluation)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.num_trials = get_config_or_default(config=self.config, config_key=NO_OF_TRIALS, default=None)
        self.k_value = get_config_or_default(config=self.config, config_key=K_VALUE, default=self.num_trials)
        
        validate_inputs_for_pass_k_initialisation(k_value=self.k_value, num_trials=self.num_trials)


    def compute(self, success_scores: List[NumericalScore]) -> NumericalScore:
        """
        Computes the pass@k metric given a list of success scores from multiple trials.

        The pass@k metric represents the probability that at least one of `k` randomly
        selected trials is successful, based on the total number of trials and the
        number of successful trials observed.

        Configuration values are retrieved from the metric config, falling back to
        defaults if not explicitly provided.

        :param success_scores: A list of :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore`
            objects, one per trial, where each score indicates success (typically 0 or 1).
        :return: A :obj:`~agent_inspect.models.metrics.metric_score.NumericalScore`
            object containing the computed pass@k value.

        :raises agent_inspect.exception.EvaluationError:
            - If ``k_value`` is less than or equal to 0
            - If ``num_trials`` is less than or equal to 0
            - If the number of provided success scores does not match ``num_trials``
            - If ``k_value`` is greater than ``num_trials``

        Example:

        >>> from agent_inspect.metrics.multi_samples import PassAtK
        >>> from agent_inspect.models.metrics import NumericalScore
        >>> from agent_inspect.metrics.constants import K_VALUE, NO_OF_TRIALS
        >>>
        >>> metric = PassAtK(config={K_VALUE: 2, NO_OF_TRIALS: 5})
        >>> scores = [NumericalScore(score=1), NumericalScore(score=0),
        ...           NumericalScore(score=1), NumericalScore(score=0),
        ...           NumericalScore(score=0)]
        >>> result = metric.compute(scores)
        >>> print(result.score)
        """
        
        num_trials = self.num_trials
        k_value = self.k_value
        
        success_scores_list = [obj.score for obj in success_scores]
        
        if num_trials > len(success_scores_list):
                raise EvaluationError(ErrorCode.INVALID_VALUE.value, f"Success scores should have the same length as num_trials ({num_trials}), but got {len(success_scores_list)}")
            
        success_count = sum(success_scores_list)

        if success_count == 0:
            return NumericalScore(score=0.0)

        if (num_trials - success_count) < k_value:
            return NumericalScore(score=1.0)

        value = 1.0 - (
            math.comb(num_trials - success_count, k_value)
            / math.comb(num_trials, k_value)
        )
        return NumericalScore(score=value)
