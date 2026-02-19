from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from agent_inspect.models.metrics.validation_result import ValidationResult

@dataclass
class NumericalScore:
    """
    Represents a numerical score produced by a metric after computation.
    """

    score: float
    """
    Contains a numerical representation of the final score after calculated by the metric.
    """
    sub_scores: Optional[Dict[str, float]] = None
    """
    Contains a dictionary of intermediate scores used to compute the final score.
    """
    explanations: Optional[List[Any]] = None
    """
    Contains a list of explanation/reason(s) for why/how this particular score is calculated.
    """
    validation_results: Optional[List[ValidationResult]] = None
    """
    Contains a list of validation results for each subgoal associated with this score.
    """

@dataclass
class BooleanScore:
    """
    Represents a boolean score produced by a metric after computation.
    """

    score: bool
    """
    Contains a boolean representation of the final score after calculated by the metric.
    """
    explanations: Optional[List[Any]] = None
    """
    Contains a list of explanation/reason(s) for why/how this particular score is calculated.
    """

