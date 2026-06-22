from dataclasses import dataclass, field
from typing import List, Dict, Optional, TypeVar, Generic

from agent_inspect.models.metrics.validation_result import (
    SubGoalValidationResult,
    ToolCallValidationResult,
)


@dataclass
class ErrorAnalysisDataSample:
    """Base class for error analysis data samples.

    Attributes:
        data_sample_id: Unique identifier for the data sample

        agent_run_id: Optional unique identifier for the agent run
    """

    data_sample_id: int
    """
    Unique identifier for the data sample. Final results of error analysis will reference this id.
    """
    agent_run_id: Optional[int] = None
    """
    Unique identifier for the agent run associated with this data sample.
    """


@dataclass
class SubgoalErrorAnalysisDataSample(ErrorAnalysisDataSample):
    """
    Represents a single data sample with its associated subgoal validations.

    Note: subgoal_validations is technically optional with an empty default to satisfy
    dataclass inheritance constraints, but in practice should always be provided.
    """

    subgoal_validations: List[SubGoalValidationResult] = field(default_factory=list)
    """
    A list of ordered subgoal validation results to perform error analysis on.
    """


@dataclass
class ToolCallErrorAnalysisDataSample(ErrorAnalysisDataSample):
    """
    Represents a single data sample with its associated tool call validations.

    Note: tool_call_validations is technically optional with an empty default to satisfy
    dataclass inheritance constraints, but in practice should always be provided.
    """

    tool_call_validations: List[ToolCallValidationResult] = field(default_factory=list)
    """
    A list of tool call validation results to perform error analysis on.
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


# Base classes for analyzed validation results


@dataclass
class AnalyzedValidation:
    """Base class for analyzed validation results.

    Attributes:
        data_sample_id: The unique identifier of the data sample

        base_error: A description of the identified error

        agent_run_id: Optional unique identifier for the agent run
    """

    data_sample_id: int
    """
    The unique identifier of the data sample that this result originates from.
    """
    base_error: Optional[str]
    """
    A description of the identified error in the validation.
    """
    agent_run_id: Optional[int] = field(default=None, kw_only=True)
    """
    Unique identifier for the agent run associated with this data sample.
    """


@dataclass
class AnalyzedSubgoalValidation(AnalyzedValidation):
    """
    Represents the error analysis result for a single subgoal validation within a data sample.
    """

    subgoal_validation: SubGoalValidationResult
    """
    The subgoal validation result being analyzed.
    """


@dataclass
class AnalyzedToolValidation(AnalyzedValidation):
    """
    Represents the error analysis result for a single tool call validation within a data sample.
    """

    tool_call_validation: ToolCallValidationResult
    """
    The tool call validation result being analyzed.
    """


# TypeVar for generic base error analysis result
T = TypeVar("T", bound=AnalyzedValidation)


@dataclass
class ErrorAnalysisResult(Generic[T]):
    """Base class for error analysis results.

    Attributes:
        analyzed_validations_clustered_by_errors: Mapping from error cluster labels
            to lists of analyzed validations that exhibit those errors
    """

    analyzed_validations_clustered_by_errors: Dict[str, List[T]]
    """
    A mapping from error cluster labels to lists of analyzed validations that exhibit those errors.
    """


@dataclass
class SubgoalErrorAnalysisResult(ErrorAnalysisResult[AnalyzedSubgoalValidation]):
    """
    Represents the overall error analysis result for a set of data samples (subgoal-based).
    """

    completed_subgoal_validations: List[AnalyzedSubgoalValidation] = field(default_factory=list)
    """
    A list of analyzed subgoal validations consisting of the subgoal validations
    that were marked completed, and thus have no associated errors.
    """


@dataclass
class ToolCallErrorAnalysisResult(ErrorAnalysisResult[AnalyzedToolValidation]):
    """
    Represents the overall error analysis result for a set of tool validation data samples.

    This result only contains failed validations clustered by error type.
    Passed validations are not included as they don't require error analysis.
    """

    pass
