from dataclasses import dataclass

from agent_inspect.models.metrics.validation_result import SubGoalValidationResult
from typing import List, Dict, Optional

@dataclass
class ErrorAnalysisDataSample:
    """
    Represents a single data sample with its associated subgoal validations.
    """

    data_sample_id: int
    """
    Unique identifier for the data sample. Final results of error analysis will reference this id.
    """
    subgoal_validations: List[SubGoalValidationResult]
    """
    A list of ordered subgoal validation results to perform error analysis on.
    """
    agent_run_id: Optional[int] = None
    """
    Unique identifier for the agent run associated with this data sample.
    """
    
@dataclass
class StatisticAnalysisResult:
    """
    Represents the statistical analysis result for a single data sample.
    """

    data_sample_id: int
    """
    The unique identifier of the data sample that this statistic analysis result corresponds to.
    """
    subgoal_validations: List[SubGoalValidationResult]
    """
    A list of subgoal validation results to perform error analysis on.
    """
    judge_expectation: Optional[float] = None
    """
    The computed expectation (mean) of judge scores across all subgoals in one data sample.
    """
    judge_std: Optional[float] = None
    """
    The computed standard deviation of judge scores across all subgoals in one data sample.
    """
    agent_run_id: Optional[int] = None
    """
    Unique identifier for the agent run associated with this data sample.
    """

@dataclass
class AnalyzedSubgoalValidation:
    """
    Represents the error analysis result for a single subgoal validation within a data sample.
    """

    subgoal_validation: SubGoalValidationResult
    """
    The subgoal validation result being analyzed.
    """
    data_sample_id: int
    """
    The unique identifier of the data sample that this result's subgoal validation originates from.
    """
    base_error: Optional[str]
    """
    A description of the identified error in the subgoal validation.
    """
    agent_run_id: Optional[int] = None
    """
    Unique identifier for the agent run associated with this data sample.
    """
    
@dataclass
class ErrorAnalysisResult:
    """
    Represents the overall error analysis result for a set of data samples.
    """

    analyzed_validations_clustered_by_errors: Dict[str, List[AnalyzedSubgoalValidation]]
    """
    A mapping from generalized errors to lists of analyzed subgoal validations that exhibit those errors.
    """
    completed_subgoal_validations: List[AnalyzedSubgoalValidation]
    """
    A list of analyzed subgoal validations consisting of the subgoal validations 
    that were marked completed, and thus have no associated errors.
    """